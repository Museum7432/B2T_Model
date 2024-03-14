#!/bin/bash



mkdir dataset

wget -P ./dataset/ https://datadryad.org/stash/downloads/file_stream/2357851

tar -xf ./dataset/competitionData.tar.gz -C ./dataset


pip install -r requirements.txt