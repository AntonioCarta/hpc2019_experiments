FOLDER=$1
NAME=$2

ch-build -t $NAME --network=host $FOLDER
ch-builder2tar $NAME $FOLDER
ch-tar2dir ./"$FOLDER"/"${NAME////.}".tar.gz $FOLDER