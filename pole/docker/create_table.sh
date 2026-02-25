sleep 15
echo "Creating database '$BC_TABLE_NAME'"
cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD "create database $BC_TABLE_NAME;"
echo "Database created!"