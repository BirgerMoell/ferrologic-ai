## Getting started with the server
Try out the server.

1. Start the server by typing 
server.py

2. Make a post request by opening a new terminal window and typing the following
curl -X POST -F 'image=@dog.jpg' http://127.0.0.1:5000/predict
