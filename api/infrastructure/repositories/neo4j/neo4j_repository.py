
from typing import Any, Dict

from neo4j import AsyncDriver
from api.infrastructure.repositories.neo4j.neo4j_utils import default_neo4jdb
from api.infrastructure.repositories.neo4j.neo4j_driver import kg_driver
from api.utils.helpers import HTTPLoggedException


class KGRepository:
    def __init__(self, driver: AsyncDriver | None = None, db_name: str = default_neo4jdb):
        self._driver = driver
        self._db_name = db_name

    @property
    def driver(self) -> AsyncDriver:
        return self._driver or kg_driver() 

    def neo4j_is_ready(self):
        return self.driver.execute_query("RETURN 'Neo4j is Ready!' as message")
    
    def drop_neo4j_indexes(self):
        """Drops and constraints and indexes present on the neo4j graph database

        Returns:
            Success or an error.
        """
        # remove all constraints
        list_constraints = self.driver.execute_query(
            """SHOW CONSTRAINTS YIELD name"""
        )
        if (list_constraints == "error"):
            return list_constraints
        constraint_names = [row["name"] for row in list_constraints["query_result"]]
        for constraint_name in constraint_names:
            dropped_constraint = self.driver.execute_query("""DROP CONSTRAINT $constraint_name""", {"constraint_name": constraint_name})
            if (dropped_constraint["status"] == "error"):
                return dropped_constraint

        # remove all indexes
        list_indexes = self.driver.execute_query(
            """SHOW INDEXES YIELD name"""
        )
        if (list_indexes == "error"):
            raise HTTPLoggedException(status_code=500, detail=f"An error has occured while listing the DB indexes: {list_indexes}")
        index_names = [row["name"] for row in list_indexes["query_result"]]
        for index_name in index_names:
            dropped_index = self.driver.execute_query("""DROP INDEX $index_name""", {"index_name": index_name})
            if (dropped_index["status"] == "error"):
                raise HTTPLoggedException(status_code=500, detail=f"An error has occured while dropping the DB index: {dropped_index}")
            
        return list_indexes
    
    def clear_neo4j_data(self) -> bool:
        """Clears all data from the neo4j graph database.

        Use with caution!

        Returns:
            True if Success 
        """
        # First, remove all nodes and relationships in batches
        data_removed = self.driver.execute_query("""MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS""")
        if (data_removed["status"] == "error") :
            raise HTTPLoggedException(status_code=500, detail=f"An error has occured while clearing the DB data: {data_removed}")

        return True

    def get_apoc_procedure_names(self):
        """List all APOC procedure names.
        APOC (Awesome Procedures on Cypher) is a library of procedures and functions that extends the capabilities of Neo4j.

        Returns:
            Success with a list of APOC procedure names or an error.
        """
        cypher = "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc' RETURN name"

        result = self.driver.execute_query(cypher)
        
        if result["status"] == "error":
            return result
        apoc_procedure_names = [row["name"] for row in result["query_result"]]

        if len(apoc_procedure_names) == 0:
            raise HTTPLoggedException(status_code=404,detail="APOC procedures not found. Please ensure APOC is installed in your Neo4j database.")
        
        return apoc_procedure_names

    def get_apoc_version(self) -> Dict[str, Any]:
        """Get the version of APOC installed in the Neo4j database.

        Returns:
            Success with the APOC version or an error.
        """
        cypher = "RETURN apoc.version() AS apoc_version"

        result = self.driver.execute_query(cypher)
        
        if result["status"] == "error":
            raise HTTPLoggedException(status_code=404,detail="An error has occured while checking the apoc version")
        
        return result["query_result"][0]["apoc_version"]
        
    def get_neo4j_version(self) -> Dict[str, Any]:
        """Get the version and edition of the Neo4j database.
        
        """
        cypher = "CALL dbms.components() yield name, versions, edition unwind versions as version return name, version, edition"

        result = self.driver.execute_query(cypher)

        return result
        

    def create_uniqueness_constraint(self, label: str,unique_property_key: str):
        """Creates a uniqueness constraint for a node label and property key.
        A uniqueness constraint ensures that no two nodes with the same label and property key have the same value.
        This improves the performance and integrity of data import and later queries.

        Args:
            label: The label of the node to create a constraint for.
            unique_property_key: The property key that should have a unique value.

        Returns:
            A dictionary with a status key ('success' or 'error').
            On error, includes an 'error_message' key.
        """    
        # Use string formatting since Neo4j doesn't support parameterization of labels and property keys when creating a constraint
        constraint_name = f"{label}_{unique_property_key}_constraint"
        query = f"""CREATE CONSTRAINT `{constraint_name}` IF NOT EXISTS
        FOR (n:`{label}`)
        REQUIRE n.`{unique_property_key}` IS UNIQUE"""
        results = self.driver.execute_query(query)
        return results