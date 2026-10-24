#!/usr/bin/env python3
"""One-shot forward shadow accumulation for an external hourly timer; never schedules itself."""
from __future__ import annotations
import json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from ops.shadow.shadow_common import SITES,raw_hourly_to_frame
from ops.shadow.shadow_predict import fetch_payload,predict_from_frame,FEED_FORWARD

def main():
    # Start one minute BEFORE the next fixed-PST issue cutoff; late results are rejected.
    issue=pd.Timestamp.now(tz='Etc/GMT+8').ceil('h');n=0
    for site in SITES:
        payload,retrieved=fetch_payload(site,issue,'forward');frame=raw_hourly_to_frame(payload);n+=len(predict_from_frame(site,issue,frame,FEED_FORWARD,created_utc=retrieved,append=True))
    print(json.dumps({'shadow_only':True,'issue_time_local':issue.isoformat(),'private_records':n}))
if __name__=='__main__':main()
