docker build -t digits:v1 -f ./docker/Dockerfile .
docker run --volume ./models:/digits/models  digits:v1