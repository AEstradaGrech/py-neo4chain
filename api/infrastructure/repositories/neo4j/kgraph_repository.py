

from typing import Any, Dict

from neo4j import AsyncDriver
from api.infrastructure.repositories.neo4j.statics import default_neo4jdb
from api.infrastructure.repositories.neo4j.neo4j_repository import Neo4jRepository
from api.utils.helpers import HTTPLoggedException


class KGraphRepository(Neo4jRepository):

    def __init__(self, driver: AsyncDriver | None = None, db_name: str = default_neo4jdb):
        super.__init__(driver, db_name)

    def load_nodes_from_csv(self, source_file: str, label: str,unique_column_name: str, properties: list[str]) -> Dict[str, Any]:
        """Batch loading of nodes from a CSV file"""

        # load nodes from CSV file by merging on the unique_column_name value
        query = f"""LOAD CSV WITH HEADERS FROM "file:///" + $source_file AS row
CALL (row) {{
    MERGE (n:$($label) {{ {unique_column_name} : row[$unique_column_name] }})
    FOREACH (k IN $properties | SET n[k] = row[k])
}} IN TRANSACTIONS OF 1000 ROWS
"""

        results =self.driver.execute_query(query, {
            "source_file": source_file,
            "label": label,
            "unique_column_name": unique_column_name,
            "properties": properties
        })
        return results

    def import_nodes(self, node_construction: dict) -> dict:
        """Import nodes as defined by a node construction rule."""

        # create a uniqueness constraint for the unique_column
        uniqueness_result = self.create_uniqueness_constraint(
            node_construction["label"],
            node_construction["unique_column_name"]
        )

        if (uniqueness_result["status"] == "error"):
            return uniqueness_result

        # import nodes from csv
        load_nodes_result = self.load_nodes_from_csv(
            node_construction["source_file"],
            node_construction["label"],
            node_construction["unique_column_name"],
            node_construction["properties"]
        )

        return load_nodes_result

    def import_relationships(self, relationship_construction: dict) -> Dict[str, Any]:
        """Import relationships as defined by a relationship construction rule."""

        # load nodes from CSV file by merging on the unique_column_name value 
        from_node_column = relationship_construction["from_node_column"]
        to_node_column = relationship_construction["to_node_column"]
        query = f"""LOAD CSV WITH HEADERS FROM "file:///" + $source_file AS row
CALL (row) {{
    MATCH (from_node:$($from_node_label) {{ {from_node_column} : row[$from_node_column] }}),
        (to_node:$($to_node_label) {{ {to_node_column} : row[$to_node_column] }} )
    MERGE (from_node)-[r:$($relationship_type)]->(to_node)
    FOREACH (k IN $properties | SET r[k] = row[k])
}} IN TRANSACTIONS OF 1000 ROWS
        """
        
        results = self.driver.execute_query(query, {
            "source_file": relationship_construction["source_file"],
            "from_node_label": relationship_construction["from_node_label"],
            "from_node_column": relationship_construction["from_node_column"],
            "to_node_label": relationship_construction["to_node_label"],
            "to_node_column": relationship_construction["to_node_column"],
            "relationship_type": relationship_construction["relationship_type"],
            "properties": relationship_construction["properties"]
        })
        return results

    def construct_domain_graph(self, construction_plan: dict) -> Dict[str, Any]:
        """Construct a domain graph according to a construction plan."""
        # first, import nodes
        node_constructions = [value for value in construction_plan.values() if value['construction_type'] == 'node']
        for node_construction in node_constructions:
            self.import_nodes(node_construction)

        # second, import relationships
        relationship_constructions = [value for value in construction_plan.values() if value['construction_type'] == 'relationship']
        for relationship_construction in relationship_constructions:
            self.import_relationships(relationship_construction)

    # Query the Neo4j database to find all nodes that have the __Entity__ label (entities created by the knowledge graph builder) and return their distinct label combinations.
    def get_entity_labels(self):
        results = self.driver.execute_query("""MATCH (n)
WHERE n:`__Entity__`
RETURN DISTINCT labels(n) AS entity_labels
    """)
        return results['query_result']

    # Flatten the label arrays into individual label strings using UNWIND, which transforms the array of labels into separate rows for each label.
    def get_unwinded_entity_labels(self):
        # unwind those lists of labels
        results = self.driver.execute_query("""MATCH (n)
WHERE n:`__Entity__`
WITH DISTINCT labels(n) AS entity_labels
UNWIND entity_labels AS entity_label
RETURN DISTINCT entity_label
            """)

        return results['query_result']

    #Filter out internal Neo4j labels that start with double underscores ("__") to focus only on the meaningful entity type labels extracted from the text.

    def filtered_labels(self):
        results = self.driver.execute_query("""MATCH (n)
WHERE n:`__Entity__`
WITH DISTINCT labels(n) AS entity_labels
UNWIND entity_labels AS entity_label
WITH entity_label
WHERE NOT entity_label STARTS WITH "__"
RETURN entity_label
        """)
        return results['query_result']
    
    #Combine the previous query steps into a reusable function that returns all unique entity labels from the subject graph, excluding internal Neo4j system labels.
    def find_unique_entity_labels(self):
        result = self.driver.execute_query("""MATCH (n)
WHERE n:`__Entity__`
WITH DISTINCT labels(n) AS entity_labels
UNWIND entity_labels AS entity_label
WITH entity_label
WHERE NOT entity_label STARTS WITH "__"
RETURN collect(entity_label) as unique_entity_labels
        """)
        if result['status'] == 'error':
            raise Exception(result['message'])
        return result['query_result'][0]['unique_entity_labels']

    #Create a function to find all unique property keys for entities of a specific label in the subject graph. This helps identify what properties are available for matching with domain graph nodes.
    def find_unique_entity_keys(self, entityLabel:str):
        result = self.driver.execute_query("""MATCH (n:$($entityLabel))
WHERE n:`__Entity__`
WITH DISTINCT keys(n) as entityKeys
UNWIND entityKeys as entityKey
RETURN collect(distinct(entityKey)) as unique_entity_keys
        """, {
            "entityLabel": entityLabel
        })
        if result['status'] == 'error':
            raise Exception(result['message'])
        return result['query_result'][0]['unique_entity_keys']

    #Create a function to find unique property keys for nodes of a specific label in the domain graph (nodes without the __Entity__ label). This enables comparison with subject graph properties for entity resolution.
            
    def find_unique_domain_keys(self, domainLabel:str):
        result = self.driver.execute_query("""MATCH (n:$($domainLabel))
WHERE NOT n:`__Entity__` // exclude entities created by the KG builder, these should be domain nodes
WITH DISTINCT keys(n) as domainKeys
UNWIND domainKeys as domainKey
RETURN collect(distinct(domainKey)) as unique_domain_keys
        """, {
            "domainLabel": domainLabel
        })
        if result['status'] == 'error':
            raise Exception(result['message'])
        return result['query_result'][0]['unique_domain_keys']

    def correlate_subject_and_domain_nodes(self, label: str, entity_key: str, domain_key: str, similarity: float = 0.9) -> dict:
        """Correlate entity and domain nodes based on label, entity key, and domain key,
        where the corresponding values of the entity and domain properties are similar
        
        For example, if you have a label "Person" and an entity key "name", and a domain key "person_name",
        this function will create a relationship like:
        (:Person:`__Entity__` {name: "John"})-[:CORRELATES_TO]->(:Person {person_name: "John"}) 
        

        Args:
            label (str): The label of the entity and domain nodes.
            entity_key (str): The key of the entity node.
            domain_key (str): The key of the domain node.
            similarity (float, optional): The similarity threshold for correlation. Defaults to 0.9.
        
        Returns:
            dict: A dictionary containing the correlation between the entity and domain nodes.
        """
        results = self.driver.execute_query("""
        MATCH (entity:$($entityLabel):`__Entity__`),(domain:$($entityLabel))
        WHERE apoc.text.jaroWinklerDistance(entity[$entityKey], domain[$domainKey]) < $distance
        MERGE (entity)-[r:CORRESPONDS_TO]->(domain)
        ON CREATE SET r.created_at = datetime() // MERGE sub-clause when the relationship is newly created
        ON MATCH SET r.updated_at = datetime()  // MERGE sub-clause when the relationship already exists
        RETURN $entityLabel as entityLabel, count(r) as relationshipCount
        """, {
            "entityLabel": label,
            "entityKey": entity_key,
            "domainKey": domain_key,
            "distance": (1.0 - similarity)
        })

        if results['status'] == 'error':
            raise HTTPLoggedException(status_code=500, detail=f"An error has occured while correlating subject and domain nodes {results['message']}")

        return results['query_result']