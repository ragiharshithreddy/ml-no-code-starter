import requests
import time
import streamlit as st

def mock_get(url):
    print("MOCK GET CALLED")
    class R:
        def raise_for_status(self): pass
        content = b'mock data'
    time.sleep(1)
    return R()

requests.get = mock_get

# ... rest of mock app?
