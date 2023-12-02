docker build -t dependency-image:V1 -f DockerfileDependency .
docker build -t final-image:v1 -f DockerfileFinalImage .
az acr build --registry rajeshiitj --image dependency-image:V1 .
az acr build --registry rajeshiitj --image final-image:v1 .