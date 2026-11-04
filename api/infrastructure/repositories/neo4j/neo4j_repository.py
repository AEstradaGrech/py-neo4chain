
from typing import Any, Dict

from loguru import logger
from neo4j import AsyncDriver
from neo4j.exceptions import DriverError, Neo4jError
from api.infrastructure.repositories.neo4j.statics import default_neo4jdb
from api.infrastructure.repositories.neo4j.neo4j_utils import kg_driver
from api.utils.helpers import HTTPLoggedException


class Neo4jRepository:
    def __init__(self, driver: AsyncDriver | None = None, db_name: str = default_neo4jdb):
        self._driver = driver
        self._db_name = db_name

    @property
    def driver(self) -> AsyncDriver:
        return self._driver or kg_driver() 

    async def neo4j_is_ready(self):
        return await self.send_query("RETURN 'Neo4j is Ready!' as message")

    async def send_query(self, cypher: str, parameters: dict | None = None) -> Dict[str, Any]:
        """Runs a query in an implicit (auto-commit) transaction, so 'CALL {...} IN TRANSACTIONS' works.

        Returns:
            {"status": "success", "query_result": [row dicts]} or {"status": "error", "error_message": str}
        """
        try:
            async with self.driver.session(database=self._db_name) as session:
                result = await session.run(cypher, parameters or {})
                records = await result.data()   # list[dict]; consume it inside the session
            return {"status": "success", "query_result": records}
        except (Neo4jError, DriverError) as e:
            logger.error(f"Cypher query failed: {e}")
            return {"status": "error", "error_message": str(e)}
        
    async def drop_neo4j_indexes(self) -> Dict[str, Any]:
        """Drops all the constraints and indexes present on the neo4j graph database, except the LOOKUP indexes:
        Neo4j creates them by default to resolve 'MATCH (n:Label)' and relationship type lookups.

        Returns:
            {"status": "success", "dropped_constraints": [names], "dropped_indexes": [names]}
            or the error of the query that failed.
        """
        # remove all constraints first: dropping a constraint also drops the index that backs it
        list_constraints = await self.send_query(
            """SHOW CONSTRAINTS YIELD name"""
        )
        if (list_constraints["status"] == "error"):
            return list_constraints
        constraint_names = [row["name"] for row in list_constraints["query_result"]]
        for constraint_name in constraint_names:
            dropped_constraint = await self.send_query("""DROP CONSTRAINT $constraint_name IF EXISTS""", {"constraint_name": constraint_name})
            if (dropped_constraint["status"] == "error"):
                return dropped_constraint

        # remove the remaining indexes (vector, fulltext, the ones created by hand...), keeping the LOOKUP ones
        list_indexes = await self.send_query(
            """SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name"""
        )
        if (list_indexes["status"] == "error"):
            return list_indexes
        index_names = [row["name"] for row in list_indexes["query_result"]]
        for index_name in index_names:
            dropped_index = await self.send_query("""DROP INDEX $index_name IF EXISTS""", {"index_name": index_name})
            if (dropped_index["status"] == "error"):
                return dropped_index

        return {"status": "success", "dropped_constraints": constraint_names, "dropped_indexes": index_names}

    async def clear_neo4j_data(self, is_full_delete:bool = False) -> bool:
        """Clears all data from the neo4j graph database.

        Use with caution!

        Args:
            is_full_delete: also drops the constraints and indexes (except the LOOKUP ones), so their labels stop
                being listed by 'db.labels()'. Property keys are never removed: that requires recreating the database.

        Returns:
            True if Success
        """
        logger.warning("DELETING ALL DATA" + (" AND CONSTRAINTS/INDEXES" if is_full_delete else ""))
        # remove all nodes and relationships in batches
        data_removed = await self.send_query("""MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS""")
        if (data_removed["status"] == "error") :
            raise HTTPLoggedException(status_code=500, detail=f"An error has occured while clearing the DB data: {data_removed}")

        if is_full_delete:
            schema_removed = await self.drop_neo4j_indexes()
            if (schema_removed["status"] == "error"):
                raise HTTPLoggedException(status_code=500, detail=f"An error has occured while dropping the DB constraints and indexes: {schema_removed}")
        return True

    async def get_apoc_procedure_names(self):
        """List all APOC procedure names.
        APOC (Awesome Procedures on Cypher) is a library of procedures and functions that extends the capabilities of Neo4j.

        Returns:
            Success with a list of APOC procedure names or an error.
        """
        cypher = "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc' RETURN name"

        result = await self.send_query(cypher)
        
        if result["status"] == "error":
            return result
        apoc_procedure_names = [row["name"] for row in result["query_result"]]

        if len(apoc_procedure_names) == 0:
            raise HTTPLoggedException(status_code=404,detail="APOC procedures not found. Please ensure APOC is installed in your Neo4j database.")
        
        return apoc_procedure_names

    async def get_apoc_version(self) -> Dict[str, Any]:
        """Get the version of APOC installed in the Neo4j database.

        Returns:
            Success with the APOC version or an error.
        """
        cypher = "RETURN apoc.version() AS apoc_version"

        result = await self.send_query(cypher)
        
        if result["status"] == "error":
            raise HTTPLoggedException(status_code=404,detail="An error has occured while checking the apoc version")
        
        return result["query_result"][0]["apoc_version"]
        
    async def get_neo4j_version(self) -> Dict[str, Any]:
        """Get the version and edition of the Neo4j database.
        
        """
        cypher = "CALL dbms.components() yield name, versions, edition unwind versions as version return name, version, edition"

        result = await self.send_query(cypher)

        return result
        

    async def create_uniqueness_constraint(self, label: str,unique_property_key: str):
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
        results = await self.send_query(query)
        return results