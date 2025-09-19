## Server side for chrome extension (PhishDetector)
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import ssl

# AI
import pickle
from logisticRegression import LogisticRegression
from functionDefs import vectorize as extract_features  # renamed from vectorize

model = None


class HTTPRequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        data = self.rfile.read(content_length)
        json_data = json.loads(data.decode())
        message = json_data.get("message", "")

        emailData = json.loads(message)
        email_sender = emailData.get("from", "")
        subject = emailData.get("subject", "")
        email_body = emailData.get("body", "")

        # If message does not contain email data, respond with error message
        if not email_sender:
            print("Data not received. Please enable Gmail original view to analyze.")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {"message": "Please enable Gmail original view to analyze"}
            self.wfile.write(json.dumps(response).encode())
            return

        print("Sender:", email_sender)
        print("Subject:", subject)
        print("Raw text:", email_body)

        result = evaluate_email(email_body, model)

        # Send response with CORS headers
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Add JSON response
        response = {"message": str(result) + "%"}
        self.wfile.write(json.dumps(response).encode())

    def do_GET(self):
        # Handle GET to avoid 501 error
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Server is running. Use POST to send email data.")


def run(server_class=HTTPServer, handler_class=HTTPRequestHandler):
    server_address = ("", 8443)
    httpd = server_class(server_address, handler_class)
    
    # Python 3.13+ compatible SSL
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print("Starting httpd on https://127.0.0.1:8443/")
    httpd.serve_forever()


def load_model():
    global model
    with open("logistic_regression_model.pkl", "rb") as file:
        model = pickle.load(file)
    print("Model loaded...")


def evaluate_email(email_body, model):
    print("Analyzing email for phishing...")
    feature_vector = extract_features(email_body)
    probability = model.predict_proba(feature_vector)
    print("Email evaluated...")
    print("Probability:", probability[0])
    return round(probability[0][1] * 100, 2)


if __name__ == "__main__":
    load_model()
    run()
