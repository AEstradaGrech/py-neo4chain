from rapidfuzz import fuzz

# for a given label, get pairs of entity and domain keys that correlate
from api.infrastructure.repositories.neo4j.kgraph_repository import KGraphRepository
from api.infrastructure.repositories.neo4j.neo4j_utils import normalize_key

class KGraphService:
    kg_repo: KGraphRepository = None
    def __init__(self):
        self.kg_repo = KGraphRepository()        

    def correlate_entity_and_domain_keys(label: str, entity_keys: list[str], domain_keys: list[str], similarity: float = 0.9) -> list[tuple[str, str]]:
        correlated_keys = []
        for entity_key in entity_keys:
            for domain_key in domain_keys:
                # only consider exact matches. this could use fuzzy matching
                normalized_entity_key = normalize_key(label, entity_key)
                normalized_domain_key = normalize_key(label, domain_key)
                # rapidfuzz similarity is 0.0 -> 100.0, so divide by 100 for 0.0 -> 1.0
                fuzzy_similarity = (fuzz.ratio(normalized_entity_key, normalized_domain_key) / 100)
                if (fuzzy_similarity > similarity): 
                    correlated_keys.append((entity_key, domain_key, fuzzy_similarity))
        correlated_keys.sort(key=lambda x: x[2], reverse=True)
        return correlated_keys

    def correlate_graph(self, similarity=0.8):
        # do it all:
        # - loop over all entity labels
        # - correlate the keys
        # - correlate (and connect) the nodes
        for entity_label in self.kg_repo.find_unique_entity_labels():
            print(f"Correlating entities labeled {entity_label}...")
            
            entity_keys = self.kg_repo.find_unique_entity_keys(entity_label)
            domain_keys = self.kg_repo.find_unique_domain_keys(entity_label)

            correlated_keys = self.correlate_entity_and_domain_keys(entity_label, entity_keys, domain_keys, similarity=similarity)

            if (len(correlated_keys) > 0):
                top_correlated_keypair = correlated_keys[0]
                print("\tbased on:", top_correlated_keypair)
                self.kg_repo.correlate_subject_and_domain_nodes(entity_label, top_correlated_keypair[0], top_correlated_keypair[1])
            else:
                print("\tNo correlation found")