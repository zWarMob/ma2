import requests


def get_access_token(
    api_key: str,
    url: str = "https://iam.cloud.ibm.com/identity/token",
    headers: dict = {"Content-Type": "application/x-www-form-urlencoded"},
    data: dict = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": None,
    },
) -> str:
    """Get access token from IBM Cloud API based on user's API key."""

    data["apikey"] = api_key
    response = requests.post(url, headers=headers, data=data)
    return response.json()["access_token"]
