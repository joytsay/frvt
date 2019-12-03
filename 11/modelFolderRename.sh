echo "[COPY] copy config/ files to tmp folder"
rsync -r config/ tmpfolder
echo "[COPY] copy config/otherConfig/ files to tmp folder"
rsync -r config/otherConfig/ tmpfolder
echo "[COPY] copy otherConfig/ files to tmp folder"
rsync -r otherConfig/ tmpfolder
echo "[DELETE] ./config ./otherConfig"
sudo rm -rf ./config ./otherConfig
echo "[RENAME] ./tmpfolder to ./config"
sudo mv ./tmpfolder ./config
echo "[CHANGEMODE] ./config"
sudo chmod -R 755 config
