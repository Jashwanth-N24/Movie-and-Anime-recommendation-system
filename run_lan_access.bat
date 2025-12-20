@echo off
echo Starting Movie Recommendation System on Local Network...
echo You can access this app from other devices on the same Wi-Fi.
echo Mobile Link: Check the Sidebar in the App for the QR Code and Link.
python -m streamlit run app.py --server.address=0.0.0.0
pause
