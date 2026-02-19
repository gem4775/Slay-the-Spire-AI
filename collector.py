import socketserver
import json
import ironclad_dict
from ironclad_dict import CARD_LIBRARY
import copy
import os
from datetime import datetime


# Make a folder for incoming JSON files
SAVE_DIR = "received_json"
os.makedirs(SAVE_DIR, exist_ok=True)


class STSHandler(socketserver.StreamRequestHandler):
    def handle(self):
        # Read each line sent by the mod
        for line in self.rfile:
            try:
                # Decode bytes → string → JSON
                data = json.loads(line.decode('utf-8').strip())

                # Pretty-print received game state
                print("\n=== New Turn Data ===")
                print(json.dumps(data, indent=2))

                # ---- NEW: Save JSON to file ----
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                filename = f"json_{timestamp}.json"
                filepath = os.path.join(SAVE_DIR, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

                print(f"Saved JSON to {filepath}")

            except json.JSONDecodeError as e:
                print("Received invalid JSON:", e)


if __name__ == "__main__":
    HOST, PORT = "127.0.0.1", 9999
    print(f"Listening for SlayTheSpire mod on {HOST}:{PORT}...")
    with socketserver.TCPServer((HOST, PORT), STSHandler) as server:
        server.serve_forever()
