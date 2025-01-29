from decouple import Config


_config = Config(".env")


def load_api_key():
    return _config("WX_API_KEY")
