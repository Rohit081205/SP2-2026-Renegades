import gc
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

def _norm_team(t):
    t = str(t).lower()
    if any(x in t for x in ["sunrisers", "hyderabad"]):               return "srh"
    if any(x in t for x in ["knight riders", "kolkata"]):             return "kkr"
    if any(x in t for x in ["royal challengers", "bengaluru", "bangalore"]): return "rcb"
    if any(x in t for x in ["indians", "mumbai"]):                    return "mi"
    if any(x in t for x in ["super kings", "chennai"]):               return "csk"
    if any(x in t for x in ["rajasthan", "royals"]):                  return "rr"
    if any(x in t for x in ["capitals", "delhi"]):                    return "dc"
    if any(x in t for x in ["kings", "punjab"]):                      return "pbks"
    if any(x in t for x in ["titans", "gujarat"]):                    return "gt"
    if any(x in t for x in ["super giants", "lucknow"]):              return "lsg"
    return "others"


def _norm_venue(v):
    v = str(v).lower()
    if any(x in v for x in ["chinnaswamy", "bengaluru",
                              "eden", "rajiv", "hyderabad", "barsapara", "guwahati", "narendra","modi","ahmedabad"]):
        return "tier_high"
    if any(x in v for x in ["wankhede","mumbai",
                          "raipur","sawai","mansingh","jaipur",
                          "shaheed","veer","narayan"]):
        return "tier_high_mod"
    if any(x in v for x in ["ekana", "lucknow"]):
        return "tier_slow"
    if any(x in v for x in ["arun", "jaitley", "delhi", "chidambaram", "chennai", "chepauk","hpca",
                            "dharamshala","himachal","pradesh", "mullanpur","yadavindra"]):
        return "tier_mod"
    return "tier_avg"

class MyModel:

    def __init__(self):
        self.player_sr   = {}
        self.lgb_model   = None
        self._id_to_name = {}

        self.BAT_BIAS = {
            "rr":     1.14,
            "rcb":    1.10,
            "kkr":    0.93,
            "mi":     1.00,
            "pbks":   1.08,
            "srh":    1.12,
            "csk":    1.06,
            "gt":     1.00,
            "lsg":    0.98,
            "dc":     0.95,
            "others": 1.00,
        }

        self.BOWL_BIAS = {
            "csk":    1.025,
            "rr":     1.032,
            "gt":     1.050,
            "lsg":    1.035,
            "dc":     1.025,
            "srh":    1.004,
            "others": 1.000,
            "mi":     0.950,
            "pbks":   0.900,
            "rcb":    1.020,
            "kkr":    0.950,
        }

        self.CHASE_BIAS = {
            "rcb":    1.04,
            "csk":    1.15,
            "mi":     1.05,
            "kkr":    1.02,
            "rr":     0.98,
            "gt":     1.00,
            "pbks":   1.02,
            "srh":    0.95,
            "dc":     0.95,
            "lsg":    0.90,
            "others": 1.00,
        }

        self.venue_base = {
            "tier_high":     {1: 61, 2: 65},
            "tier_high_mod": {1: 59, 2: 62},
            "tier_mod":      {1: 54, 2: 62},
            "tier_slow":     {1: 50, 2: 47},
            "tier_avg":      {1: 57, 2: 58},
        }

        self.TEAM_BOUNDS = {
            "srh":  (20, 113),   
            "rr":   (32, 105),   
            "rcb":  (43,  90),
            "pbks": (47, 101),   
            "kkr":  (28,  86),
            "mi":   (33,  88),
            "csk":  (33,  85),
            "gt":   (37,  95),
            "lsg":  (31,  97),
            "dc":   (25,  73),
            "others":(25, 75),
        }

        self._HEU_NEG_CAP = 0.50
        self._HEU_POS_CAP = 0.35

    def _safe_parse_ids(self, raw):
        ids = []
        if not raw or (isinstance(raw, float) and np.isnan(raw)):
            return ids
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                val = int(float(token))
                if val >= 0:
                    ids.append(val)
            except (ValueError, OverflowError):
                pass
        return ids

    def fit(self, deliveries_df, players_df=None, matches_df=None):
        if deliveries_df is None or deliveries_df.empty:
            return self

        try:
            if players_df is not None and not players_df.empty:
                id_col   = next((c for c in ['player_id', 'id', 'ID', 'Player_Id']
                                 if c in players_df.columns), None)
                name_col = next((c for c in ['player_name', 'name', 'Name', 'Player_Name']
                                 if c in players_df.columns), None)
                if id_col and name_col:
                    for _, r in players_df[[id_col, name_col]].dropna().iterrows():
                        try:
                            pid = int(float(r[id_col]))
                            if pid >= 0:
                                self._id_to_name[pid] = str(r[name_col]).strip()
                        except (ValueError, OverflowError):
                            pass

            if matches_df is not None and not matches_df.empty:
                merge_cols = [c for c in ['matchId', 'venue', 'date']
                              if c in matches_df.columns]
                deliveries_df = deliveries_df.merge(
                    matches_df[merge_cols], on='matchId', how='left'
                )
                if 'date' in matches_df.columns:
                    date_strs  = matches_df['date'].astype(str)
                    recent_ids = matches_df.loc[
                        date_strs >= '2022-01-01', 'matchId'
                    ].unique()
                    deliveries_df = deliveries_df[
                        deliveries_df['matchId'].isin(recent_ids)
                    ]

            if 'venue' not in deliveries_df.columns or deliveries_df['venue'].isna().all():
                try:
                    m = pd.read_csv("/app/training_data/matches_updated_ipl_upto_2025.csv")
                    mc = 'matchId' if 'matchId' in m.columns else 'id'
                    vmap = m.set_index(mc)['venue'].to_dict()
                    deliveries_df = deliveries_df.copy()
                    deliveries_df['venue'] = deliveries_df['matchId'].map(vmap)
                except Exception:
                    deliveries_df = deliveries_df.copy()
                    deliveries_df['venue'] = 'unknown'

            pp = deliveries_df[deliveries_df['over'] < 6].copy()

            if 'extras' in pp.columns:
                pp['total_runs'] = pp['batsman_runs'] + pp['extras']
            elif 'extra_runs' in pp.columns:
                pp['total_runs'] = pp['batsman_runs'] + pp['extra_runs']
            else:
                pp['total_runs'] = pp['batsman_runs']

            self.player_sr = (
                pp.groupby('batsman')['total_runs'].sum()
                / pp.groupby('batsman')['matchId'].nunique()
            ).to_dict()

            required_group_cols = ['matchId', 'inning']
            optional_group_cols = [c for c in ['venue', 'batting_team', 'bowling_team']
                                   if c in pp.columns]
            all_group_cols = required_group_cols + optional_group_cols

            train_rows = []
            for keys, g in pp.groupby(all_group_cols):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                keys = dict(zip(all_group_cols, keys))
                inn  = int(keys['inning'])
                ven  = keys.get('venue', '')
                bat  = keys.get('batting_team', 'others')
                bowl = keys.get('bowling_team', 'others')

                vt        = _norm_venue(ven)
                bk        = _norm_team(bat)
                bwl_k     = _norm_team(bowl)
                base      = self.venue_base.get(vt, self.venue_base['tier_avg']).get(inn, 58)
                bm        = self.BAT_BIAS.get(bk, 1.0)
                bwm       = 1.0 / self.BOWL_BIAS.get(bwl_k, 1.0)
                cm        = self.CHASE_BIAS.get(bk, 1.0) if inn == 2 else 1.0
                pred_base = base * bm * bwm * cm
                train_rows.append({
                    'inn':  inn,
                    'mult': bm * bwm * cm,
                    'base': base,
                    'res':  g['total_runs'].sum() - pred_base,
                })

            tdf = pd.DataFrame(train_rows)
            self.lgb_model = lgb.train(
                {'objective': 'regression', 'metric': 'mae',
                 'verbosity': -1, 'learning_rate': 0.05},
                lgb.Dataset(tdf[['inn', 'mult', 'base']], label=tdf['res']),
                num_boost_round=300,
            )
            del tdf

        except Exception as e:
            print(f"[fit] error: {e}")
        finally:
            del deliveries_df
            gc.collect()

        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        results = []

        for _, row in test_df.iterrows():
            try:
                vt       = _norm_venue(str(row.get('venue', '')))
                inn      = int(row.get('innings', row.get('inning', 1)))
                bat_team = _norm_team(str(row.get('batting_team', '')))
                bwl_team = _norm_team(str(row.get('bowling_team', '')))

                base   = self.venue_base.get(vt, self.venue_base['tier_avg']).get(inn, 58)
                bm     = self.BAT_BIAS.get(bat_team, 1.0)
                bwm    = 1.0 / self.BOWL_BIAS.get(bwl_team, 1.0)
                cm     = self.CHASE_BIAS.get(bat_team, 1.0) if inn == 2 else 1.0
                hat    = base * bm * bwm * cm

                lgb_adj = 0.0
                if self.lgb_model is not None:
                    lgb_res = float(self.lgb_model.predict(
                        np.array([[inn, bm * bwm * cm, base]], dtype=np.float32)
                    )[0])
                    lgb_adj = 0.15 * np.clip(lgb_res, -15.0, 50.0)

                p_ids   = self._safe_parse_ids(row.get("Batsman's Player Id", ""))
                d_count = len(p_ids)

                wkt = min(8, max(0, d_count - 2)) 
                wkt_delta = {  
                    0: min(10, base * 0.15),
                    1: +3,    
                    2: -6,    
                    3: -10,   
                    4: -18,   
                    5: -23,   
                    6: -27,   
                    7: -29,
                    8: -30,
                }
                depth_adj = wkt_delta[wkt]

                is_explosive = bat_team in {'srh', 'rr', 'rcb', 'mi', 'pbks', 'kkr'}
                bench   = 9.5 if is_explosive else 8.5
                sr_adj  = 0.0
                for pid in p_ids[:3]:
                    if pid in self.player_sr:
                        sr_adj += (self.player_sr[pid] - bench) * 0.2

                total_heu = depth_adj + sr_adj
                total_heu = np.clip(total_heu,
                                    -self._HEU_NEG_CAP * hat,
                                     self._HEU_POS_CAP * hat)

                score = hat + lgb_adj + total_heu

                lo, hi = self.TEAM_BOUNDS.get(bat_team, self.TEAM_BOUNDS["others"])
                score = max(lo, min(hi, score))

            except Exception:
                score = 55.0

            results.append({'id': row['id'], 'predicted_score': int(round(score))})

        return pd.DataFrame(results)[['id', 'predicted_score']]


if __name__ == "__main__":
    m = MyModel()
    for d_path in ("deliveries_updated_ipl_upto_2025.csv", "deliveries.csv"):
        if os.path.exists(d_path):
            m.fit(
                pd.read_csv(d_path),
                pd.read_csv("ipl_players_uniqueid.csv") if os.path.exists("ipl_players_uniqueid.csv") else None,
                pd.read_csv("matches_updated_ipl_upto_2025.csv") if os.path.exists("matches_updated_ipl_upto_2025.csv") else None,
            )
            break
    if os.path.exists("test_file.csv"):
        out = m.predict(pd.read_csv("test_file.csv"))
        out.to_csv("submission.csv", index=False)
        print(out.to_string(index=False))