# Intruder Detector UI
This is a web-based UI specifically designed to display the information that the Intruder-detector reference implementation processes.   
This web browser UI is not real-time, but uses the information procesed by the application. Because of this, the UI should be started after the application has been stopped.

## Running the UI
Make sure you are in the `UI/` folder, where the `index.html` file is.

Chrome*:
```
google-chrome --allow-file-access-from-files --allow-file-access --allow-cross-origin-auth-prompt index.html
```
Firefox*:
```
firefox index.html
```
