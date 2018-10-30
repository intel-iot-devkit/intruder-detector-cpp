# Intruder Detector UI
This is a web-based UI specifically designed to display the information that the Store-traffic-monitor reference implementation processes.   
This web browser UI is not real-time, but uses the information procesed by the application. Because of this, the UI should be started after the application has been stopped.

## Running the UI
In the `UI` folder of the reference-implementation run the following commands.   

Chrome*:
```
google-chrome  --user-data-dir=$HOME/.config/google-chrome/Intruder-detector --new-window --allow-file-access-from-files --allow-file-access --allow-cross-origin-auth-prompt index.html
```
Firefox*:
```
firefox index.html
```
**_Note:_** For Firefox*, if the alerts list does not appear on the right side of the browser window, click anywhere on video progress bar to trigger a refresh.
