# Teams Trigger to Launch Streamlit App (macOS)

This guide explains how to configure your macOS environment so that typing `/km` in Microsoft Teams (personal account) automatically launches a Streamlit application in Safari.

Sideloading MS Teams message extension (i.e. the slash command) is only possible to be done in work/school account where we need to build a message extension/bot with the Developer Portal for Teams and upload as a custom app which rquires an M365 account + admin settings. Naturally, this feature addition needs permission from Amgen MS Teams admin which typically takes a few weeks. Since the time is of the essence, the proof-of-concept (PoC) demo is done through personal laptop so org admin permission is not required. Since sideloading personal MS Teams extension is not possible, we are going to fake the experience of web application being invoked by MS Teams message extension. 

We will use a tiny launcher script + a system-wide text expander that runs scripts when you type a trigger. Specifically, we will use **espanso** (an open-source text expander) to simulate a MS Teams command trigger. When we type "/km" in MS Teams, it will start the target Streamlit app in the background and open Safari browser.

Why not other routes?
- Power Automate from a personal Teams account won't launch a process on your Mac wihout a local agent anyway, and custom app/bot upload needs an org tenant (i.e. Work/School MS Teams account) as discussed above.

At the time of writing, development for KM-specific WebApp-based ChatBot for data ingestion and intelligent search is still being actively developed. This is the reason why the demo video below shows Streamlit application running a past AI project. The intention for the video below is to capture the seamless experience of MS Teams message extension to ultimately invoke Safari browser without having the need to remember which application/environment/URL to launch. At the end of the day, the intention is to deploy "/km" capability to encourage Amgen staffs for effortless idea/asset submission to Knowledge Marketplace.

[YouTube Video LINK](https://youtu.be/4khPg8f5I4E)

[![YouTube Thumbnail](https://img.youtube.com/vi/4khPg8f5I4E/hqdefault.jpg)](https://youtu.be/4khPg8f5I4E)

------------------------------------------------------------------------

## 1. Install Prerequisites

### Install Homebrew (if not installed)

``` bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install espanso

``` bash
brew install --cask espanso
```

Start espanso from Applications, and grant **Accessibility** and **Input Monitoring** permissions under: **System Settings → Privacy & Security → Accessibility**.

Next, obtain espanso's directory can be obtained by running below command.
```bash
espanso path
```

If your installation is correct, you will get the output below.
```bash
Config: ~/Library/Application Support/espanso
Packages: ~/Library/Application Support/espanso/match/packages
Runtime: ~/Library/Caches/espanso
```

espanso's directory is `~/Library/Application Support/espanso` in this case.

Then, create a `scripts` directory inside the espanso's directory.
```bash
cd "~/Library/Application Support/espanso"
mkdir scripts
```

------------------------------------------------------------------------

## 2. Prepare Your Streamlit Script

Create a launcher script that runs your Streamlit app by placing `km_launcher.sh` in the `scripts` directory mentioned earlier.

Example: **`~/Library/Application Support/espanso/scripts/km_launcher.sh`**

Please update `ENV_NAME` and `APP_PATH` variables accordingly. 
- `ENV_NAME`: Environment variable that denotes the conda environment name in which all packages required to run the Streamlit application will be installed.
- `APP_PATH`: Environment variable that denotes the absolute path to Streamlit application file `app.py`.

Make it executable:

``` bash
chmod +x "~/Library/Application Support/espanso/scripts/km_launcher.sh"
```

**Optional:**
- `km_launcher.sh` masks the conda environment activation and Streamlit application launch. In case you want to see the sequence of processes run in the background, can replace the content of `km_launcher.sh` with `km_launcher.sh.visible_execution_in_terminal`

------------------------------------------------------------------------

## 3. Configure espanso Match

Create a match definition so typing `/km` triggers the script by placing `km_launcher.yml` in the `match` directory within espanso's directory mentioned earlier.

Example: **`~/Library/Application Support/espanso/match/km_launcher.yml`**

Reload espanso:

``` bash
espanso restart
```

Whenever you update the `km_launcher.sh` or `km_launcher.yml`, need to restart espanso in the terminal as shown above.


------------------------------------------------------------------------

## 4. Test the Setup

1.  Open Microsoft Teams (personal).
2.  Type `/km`.
3.  Safari will open with your Streamlit app.
4.  Logs:
    -   Launcher: `/tmp/km_launcher.log`
    -   Streamlit: `/tmp/km_streamlit.log`

------------------------------------------------------------------------

## 5. Troubleshooting

-   If nothing happens, ensure `{{out}}` is included in `replace:` within `~/Library/Application Support/espanso/match/km_launcher.yml` (required to execute scripts).
-   If apps still won't open, try running espanso in unmanaged mode:
``` bash
espanso service stop
espanso start --unmanaged
```
-   Confirm espanso has **Accessibility** and **Input Monitoring** permissions in macOS settings.

------------------------------------------------------------------------

## Result

Now you can demo a Teams-style "message extension" on a personal account: typing `/km` launches your local Streamlit web app in Safari.

------------------------------------------------------------------------

## Close Port

Just closing Streamlit Application in Safari won't terminate the process ID. To avoid having a process ID running in the background, need to manually kill such process ID.

```bash
lsof -iTCP:8501 -sTCP:LISTEN -n -P\n
kill -9 <pid>
```

------------------------------------------------------------------------
