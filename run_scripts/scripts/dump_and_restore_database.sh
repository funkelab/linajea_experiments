if [ $# -ne 2 ]
then
    echo "No arguments supplied"
    exit
fi

echo $1 $2
db_name=$1
db_name_out=$2

mongodump -u linajeaAdmin -p FeOOHnH2O --host rsLinajea/funke-mongodb4 -o mongoBackup -vvvv --authenticationDatabase=admin -d ${db_name} -c predict_daisy
mongodump -u linajeaAdmin -p FeOOHnH2O --host rsLinajea/funke-mongodb4 -o mongoBackup -vvvv --authenticationDatabase=admin -d ${db_name} -c nodes
# mongodump -u linajeaAdmin -p FeOOHnH2O --host rsLinajea/funke-mongodb4 -o mongoBackup -vvvv --authenticationDatabase=admin -d ${db_name} -c db_meta_info
# mongodump -u linajeaAdmin -p FeOOHnH2O --host rsLinajea/funke-mongodb4 -o mongoBackup -vvvv --authenticationDatabase=admin -d ${db_name} -c meta
mongodump -u linajeaAdmin -p FeOOHnH2O --host rsLinajea/funke-mongodb4 -o mongoBackup -vvvv --authenticationDatabase=admin -d ${db_name} -c edges
mongodump -u linajeaAdmin -p FeOOHnH2O --host rsLinajea/funke-mongodb4 -o mongoBackup -vvvv --authenticationDatabase=admin -d ${db_name} -c parameters

mongorestore -u linajeaAdmin -p FeOOHnH2O --host rsLinajea/funke-mongodb4  -vvvv --authenticationDatabase=admin -d ${db_name_out} -c predict_daisy mongoBackup/${db_name}/predict_daisy.bson
mongorestore -u linajeaAdmin -p FeOOHnH2O --host rsLinajea/funke-mongodb4  -vvvv --authenticationDatabase=admin -d ${db_name_out} -c nodes mongoBackup/${db_name}/nodes.bson
# mongorestore -u linajeaAdmin -p FeOOHnH2O --host rsLinajea/funke-mongodb4  -vvvv --authenticationDatabase=admin -d ${db_name_out} -c db_meta_info mongoBackup/${db_name}/db_meta_info.bson
# mongorestore -u linajeaAdmin -p FeOOHnH2O --host rsLinajea/funke-mongodb4  -vvvv --authenticationDatabase=admin -d ${db_name_out} -c meta mongoBackup/${db_name}/meta.bson
mongorestore -u linajeaAdmin -p FeOOHnH2O --host rsLinajea/funke-mongodb4  -vvvv --authenticationDatabase=admin -d ${db_name_out} -c edges mongoBackup/${db_name}/edges.bson
mongorestore -u linajeaAdmin -p FeOOHnH2O --host rsLinajea/funke-mongodb4  -vvvv --authenticationDatabase=admin -d ${db_name_out} -c parameters mongoBackup/${db_name}/parameters.bson
