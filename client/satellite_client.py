from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import json


class SentinelHubClient:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
        self.api_url = 'https://sh.dataspace.copernicus.eu'
        self.session = None
        self.token = None

    def authenticate(self):
        """Authenticate and retrieve an access token."""
        try:
            client = BackendApplicationClient(client_id=self.client_id)
            self.session = OAuth2Session(client=client)
            self.token = self.session.fetch_token(
                token_url=self.token_url,
                client_secret=self.client_secret,
                include_client_id=True,
            )
            print("Authentication successful!")
        except Exception as e:
            print("Error during authentication:", e)
            raise

    def get_instances(self):
        """Fetch available WMS instances."""
        url = f"{self.api_url}/configuration/v1/wms/instances"
        response = self.session.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print("Error fetching instances:", response.content.decode('utf-8'))
            return None

    def process_request(self, bbox, time_range, evalscript, width=512, height=512):
        """Send a process request to Sentinel Hub."""
        request_payload = {
            "input": {
                "bounds": {
                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                    "bbox": bbox,
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {"timeRange": time_range},
                    }
                ],
            },
            "output": {"width": width, "height": height},
            "evalscript": evalscript,
        }

        url = f"{self.api_url}/api/v1/process"
        response = self.session.post(url, json=request_payload)

        if response.status_code == 200:
            print("Processing request successful!")
            return response.content  # Returns the raw binary data
        else:
            print("Error in processing request:", response.content.decode('utf-8'))
            return None


# Example usage
if __name__ == "__main__":
    # Client credentials
    CLIENT_ID = "sh-38921ca9-27a7-4d5a-9578-80bf4e60e597"
    CLIENT_SECRET = "dmv9dpUGGrSBSDycKUPOjvCvDPUz2wGl"

    # Create the client
    sh_client = SentinelHubClient(CLIENT_ID, CLIENT_SECRET)

    # Authenticate
    sh_client.authenticate()

    # Get instances
    instances = sh_client.get_instances()
    print("Available WMS Instances:", json.dumps(instances, indent=4))

    # Evalscript and parameters
    EVALSCRIPT = """
    //VERSION=3
    function setup() {
      return {
        input: ["B02", "B03", "B04"],
        output: { bands: 3, sampleType: "AUTO" },
      };
    }
    function evaluatePixel(sample) {
      return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
    }
    """
    BBOX = [
        13.822174072265625,
        45.85080395917834,
        14.55963134765625,
        46.29191774991382,
    ]
    TIME_RANGE = {"from": "2022-10-01T00:00:00Z", "to": "2022-10-31T00:00:00Z"}

    # Process request
    image_data = sh_client.process_request(BBOX, TIME_RANGE, EVALSCRIPT)
    if image_data:
        with open("output.tif", "wb") as file:
            file.write(image_data)
        print("Image saved as 'output.tif'")