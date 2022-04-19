from text.symbols import *

ttsing_pitch_to_id = {p: i for i, p, in enumerate(ttsing_pitch_set)}
id_to_ttsing_pitch = {i: p for i, p, in enumerate(ttsing_pitch_set)}

learn2sing_pho_to_id = {p: i for i, p, in enumerate(learn2sing_phone_set)}
learn2sing_id_to_pho = {p: i for i, p, in enumerate(learn2sing_phone_set)}
