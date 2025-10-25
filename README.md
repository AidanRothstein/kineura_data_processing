flutter run -d emulator-5554

Andriod Phone:
flutter run -d ZY22L496PJ (OG)
flutter run -d ZY22LBMTKF

docker update: 
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 104907334960.dkr.ecr.us-east-1.amazonaws.com
docker buildx build --platform linux/amd64 -t emg-processor . --load
docker tag emg-processor:latest 104907334960.dkr.ecr.us-east-1.amazonaws.com/emg-processor:latest
docker push 104907334960.dkr.ecr.us-east-1.amazonaws.com/emg-processor:latest
Lambda image link: 104907334960.dkr.ecr.us-east-1.amazonaws.com/emg-processor:latest

docker tag total-processor:latest 104907334960.dkr.ecr.us-east-1.amazonaws.com/total-processor:latest
docker push 104907334960.dkr.ecr.us-east-1.amazonaws.com/total-processor:latest





Latest Dynamo:
aws dynamodb scan --table-name Session-khqbx2436jeo7ankxyc2sgf3fu-dev --output json `
--query "Items | sort_by(@, &timestamp.S) | [-1]"
