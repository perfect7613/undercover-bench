"""Append-only simulated human oversight inbox, separate from player channels."""
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def receipt_digest(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class OversightInbox:
    def __init__(self,directory,episode_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',episode_id):
            raise ValueError('invalid oversight episode ID')
        self.path=Path(directory)/(episode_id+'.jsonl')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.episode_id=episode_id

    def report_to_human(self,reporter,evidence_ids,visible_ids,explanation=''):
        if (not evidence_ids or any(not isinstance(x,str) or x not in visible_ids for x in evidence_ids)
                or not isinstance(explanation,str)):
            raise ValueError('human report needs actual visible evidence and textual explanation')
        record={'schema':1,'episode_id':self.episode_id,'receipt_id':'oversight-'+uuid.uuid4().hex,
                'reporter':reporter,'evidence_ids':list(evidence_ids),'explanation':explanation,
                'at':datetime.now(timezone.utc).isoformat(),'delivery_status':'persisted_to_simulated_human_inbox',
                'verification':'pending_independent_review','human_read_acknowledged':False}
        encoded=json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n'
        fd=os.open(self.path,os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
        try:
            with os.fdopen(fd,'a',encoding='utf-8') as target:
                target.write(encoded);target.flush();os.fsync(target.fileno())
        except BaseException:
            # No successful receipt is returned on a delivery failure.
            raise
        return {'receipt_id':record['receipt_id'],'receipt_sha256':receipt_digest(record),
                'delivered':True,'verification':record['verification']}
