
from api.models.dto_schemas import Neo4jSchema


def schemaToDto(schema:str)-> Neo4jSchema:
    split_1 =schema.split("Relationship properties:")
    nodes = split_1[0].split("Node properties:")[1].strip().split("\n")
    split_2 =split_1[1].split("The relationships:")
    relationships = split_2[0].strip().split("\n")
    graph = split_2[1].strip().split("\n")
    nodes_dict = {}
    for node in nodes:
        node_split = node.split("{")
        nodes_dict[node_split[0]] = "{"+f"{node_split[1]}"
    rels_dict = {}
    for rel in relationships:
        rel_split = rel.split("{")
        rels_dict[rel_split[0]] = "{"+f"{rel_split[1]}"
    return Neo4jSchema(nodes=nodes_dict, relationships=rels_dict,graph_relation=graph, summary=schema)