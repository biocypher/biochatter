from biocypher import BioCypher
from pole.adapters.pole_adapter import (
    PoleAdapter,
)

bc = BioCypher()

adapter = PoleAdapter()
bc.write_nodes(adapter.get_nodes())
bc.write_edges(adapter.get_edges())

# Write admin import statement
bc.write_import_call()
bc.write_schema_info(as_node=True)

# Print summary
bc.summary()

# # Ontology information
# ont = bc._get_ontology()
# print(ont._nx_graph.nodes)
