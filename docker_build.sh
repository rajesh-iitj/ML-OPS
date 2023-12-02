docker build -t digits:v1 -f ./Dockerfile .
docker run --volume ./models:/digits/models  digits:v03