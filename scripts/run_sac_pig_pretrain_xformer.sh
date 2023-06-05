#!/bin/bash

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python nanorl/sac/run_control_suite.py \
    --pretrain_envs RoboPianist-repertoire-150-ArabesqueNo1-v0 \
                    RoboPianist-repertoire-150-ArabesqueNo2-v0 \
                    RoboPianist-repertoire-150-BalladeNo1-v0 \
                    RoboPianist-repertoire-150-BalladeNo2-v0 \
                    RoboPianist-repertoire-150-Berceuse-v0 \
                    RoboPianist-repertoire-150-CarnivalOp37ANo2-v0 \
                    RoboPianist-repertoire-150-ClairDeLune-v0 \
                    RoboPianist-repertoire-150-EnglishSuiteNo2Prelude-v0 \
                    RoboPianist-repertoire-150-EnglishSuiteNo3Prelude-v0 \
                    RoboPianist-repertoire-150-EtudeOp10No12-v0 \
                    RoboPianist-repertoire-150-EtudeOp10No3-v0 \
                    RoboPianist-repertoire-150-EtudeOp25No11-v0 \
                    RoboPianist-repertoire-150-FantaisieImpromptu-v0 \
                    RoboPianist-repertoire-150-FantasieK475-v0 \
                    RoboPianist-repertoire-150-FantasieStuckeOp12No7-v0 \
                    RoboPianist-repertoire-150-ForElise-v0 \
                    RoboPianist-repertoire-150-FrenchSuiteNo3Minuet-v0 \
                    RoboPianist-repertoire-150-FrohlicherLandmannOp68No10-v0 \
                    RoboPianist-repertoire-150-GoldbergVariationsVariation13-v0 \
                    RoboPianist-repertoire-150-GrandeValseBrillanteOp18-v0 \
                    RoboPianist-repertoire-150-GymnopedieNo1-v0 \
                    RoboPianist-repertoire-150-HolborgSuiteOp40No5-v0 \
                    RoboPianist-repertoire-150-HumoreskeOp101No7-v0 \
                    RoboPianist-repertoire-150-HungarianRhapsodyNo2-v0 \
                    RoboPianist-repertoire-150-ImpromptuNo3-v0 \
                    RoboPianist-repertoire-150-ImpromptuOp90No3-v0 \
                    RoboPianist-repertoire-150-ImpromptuOp90No4-v0 \
                    RoboPianist-repertoire-150-IntermezzoOp118No2-v0 \
                    RoboPianist-repertoire-150-ItalianConverto1StMov-v0 \
                    RoboPianist-repertoire-150-JeTeVeux-v0 \
                    RoboPianist-repertoire-150-JeuxDeau-v0 \
                    RoboPianist-repertoire-150-KinderszenenOp15No1-v0 \
                    RoboPianist-repertoire-150-KreislerianaOp16No1-v0 \
                    RoboPianist-repertoire-150-KreislerianaOp16No3-v0 \
                    RoboPianist-repertoire-150-LaCampanella-v0 \
                    RoboPianist-repertoire-150-LaChasseOp19No3-v0 \
                    RoboPianist-repertoire-150-LaFilleAuxCheveuxDeLin-v0 \
                    RoboPianist-repertoire-150-LiebestraumNo3-v0 \
                    RoboPianist-repertoire-150-LyricPiecesOp12No1-v0 \
                    RoboPianist-repertoire-150-LyricPiecesOp38No1-v0 \
                    RoboPianist-repertoire-150-LyricPiecesOp43No1-v0 \
                    RoboPianist-repertoire-150-LyricPiecesOp54No3-v0 \
                    RoboPianist-repertoire-150-LyricPiecesOp62No2-v0 \
                    RoboPianist-repertoire-150-LyricPiecesOp62No4-v0 \
                    RoboPianist-repertoire-150-MapleLeafRag-v0 \
                    RoboPianist-repertoire-150-MazurkaOp7No1-v0 \
                    RoboPianist-repertoire-150-MinuetInGMajorWoo102-v0 \
                    RoboPianist-repertoire-150-MusicalMomentOp16No4-v0 \
                    RoboPianist-repertoire-150-NocturneOp9No2-v0 \
                    RoboPianist-repertoire-150-NorwegianDanceOp35No3-v0 \
                    RoboPianist-repertoire-150-PartitaNo42-v0 \
                    RoboPianist-repertoire-150-PartitaNo6Corrente-v0 \
                    RoboPianist-repertoire-150-PavanePourUneInfanteDefunte-v0 \
                    RoboPianist-repertoire-150-PeerGyntOp46No2-v0 \
                    RoboPianist-repertoire-150-PianoSonata1StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataD8453RdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK280InFMajor2NdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK281InBbMajor1StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK282InEbMajorMinuet1-v0 \
                    RoboPianist-repertoire-150-PianoSonataK283InGMajor1StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK284InDMajor1StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK284InDMajor3RdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK310InAMinor1StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK330InCMajor1StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK330InCMajor2NdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK331InAMajor3RdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK332InFMajor1StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK332InFMajor3RdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK457InCMinor3RdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK545InCMajor1StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK570InBbMajor1StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK570InBbMajor2NdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK576InDMajor2NdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataK576InDMinor1StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo141StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo142NdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo143RdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo211StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo213RdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo241StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo242NdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo281StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo282NdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo301StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo303RdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo31StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo32NdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo41StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo43RdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo5-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo51StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo81StMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo82NdMov-v0 \
                    RoboPianist-repertoire-150-PianoSonataNo83RdMov-v0 \
                    RoboPianist-repertoire-150-PicturesAtAnExhibitionBydlo-v0 \
                    RoboPianist-repertoire-150-PicturesAtAnExhibitionGreatKiev-v0 \
                    RoboPianist-repertoire-150-PicturesAtAnExhibitionPromenade-v0 \
                    RoboPianist-repertoire-150-PolonaiseFantasieOp61-v0 \
                    RoboPianist-repertoire-150-PolonaiseOp40No1-v0 \
                    RoboPianist-repertoire-150-PolonaiseOp53-v0 \
                    RoboPianist-repertoire-150-PreludeBook1No2-v0 \
                    RoboPianist-repertoire-150-PreludeOp23No5-v0 \
                    RoboPianist-repertoire-150-PreludeOp23No9-v0 \
                    RoboPianist-repertoire-150-PreludeOp28No17-v0 \
                    RoboPianist-repertoire-150-PreludeOp28No19-v0 \
                    RoboPianist-repertoire-150-PreludeOp28No7-v0 \
                    RoboPianist-repertoire-150-PreludeOp3No2-v0 \
                    RoboPianist-repertoire-150-Reverie-v0 \
                    RoboPianist-repertoire-150-RhapsodieOp79No2-v0 \
                    RoboPianist-repertoire-150-RomanianDanceNo1-v0 \
                    RoboPianist-repertoire-150-ScherzoNo2-v0 \
                    RoboPianist-repertoire-150-ScherzoNo3-v0 \
                    RoboPianist-repertoire-150-ScottJoplinsNewRag-v0 \
                    RoboPianist-repertoire-150-Sicilienne-v0 \
                    RoboPianist-repertoire-150-SinfoniaNo12-v0 \
                    RoboPianist-repertoire-150-SonataInAMajorK208-v0 \
                    RoboPianist-repertoire-150-Sonatine1StMov-v0 \
                    RoboPianist-repertoire-150-SongWithoutWordsOp19No1-v0 \
                    RoboPianist-repertoire-150-SuiteBergamasquePasspied-v0 \
                    RoboPianist-repertoire-150-SuiteBergamasquePrelude-v0 \
                    RoboPianist-repertoire-150-SuiteEspanolaOp45No1-v0 \
                    RoboPianist-repertoire-150-TheEntertainer-v0 \
                    RoboPianist-repertoire-150-TwoPartInventionInCMajor-v0 \
                    RoboPianist-repertoire-150-TwoPartInventionInCMinor-v0 \
                    RoboPianist-repertoire-150-TwoPartInventionInDMajor-v0 \
                    RoboPianist-repertoire-150-TwoPartInventionInFMajor-v0 \
                    RoboPianist-repertoire-150-VenetianischesGondelliedOp30No6-v0 \
                    RoboPianist-repertoire-150-WaltzOp39No15-v0 \
                    RoboPianist-repertoire-150-WaltzOp64No2-v0 \
                    RoboPianist-repertoire-150-WaltzOp69No1-v0 \
                    RoboPianist-repertoire-150-WaltzOp69No2-v0 \
                    RoboPianist-repertoire-150-WandererFantasy-v0 \
                    RoboPianist-repertoire-150-WellTemperedClavierBookIPreludeNo23InBMajor-v0 \
                    RoboPianist-repertoire-150-WellTemperedClavierBookIPreludeNo2InCMinor-v0 \
                    RoboPianist-repertoire-150-WellTemperedClavierBookIPreludeNo7InEbMajor-v0 \
                    RoboPianist-repertoire-150-WellTemperedClavierBookIiFugueNo19InAMajor-v0 \
                    RoboPianist-repertoire-150-WellTemperedClavierBookIiFugueNo2InCMinor-v0 \
                    RoboPianist-repertoire-150-WellTemperedClavierBookIiPreludeNo11InFMajor-v0 \
    --eval_envs RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0 \
                RoboPianist-etude-12-FrenchSuiteNo5Sarabande-v0 \
                RoboPianist-etude-12-PianoSonataD8451StMov-v0 \
                RoboPianist-etude-12-PartitaNo26-v0 \
                RoboPianist-etude-12-WaltzOp64No1-v0 \
                RoboPianist-etude-12-BagatelleOp3No4-v0 \
                RoboPianist-etude-12-KreislerianaOp16No8-v0 \
                RoboPianist-etude-12-FrenchSuiteNo5Gavotte-v0 \
                RoboPianist-etude-12-PianoSonataNo232NdMov-v0 \
                RoboPianist-etude-12-GolliwoggsCakewalk-v0 \
                RoboPianist-etude-12-PianoSonataNo21StMov-v0 \
                RoboPianist-etude-12-PianoSonataK279InCMajor1StMov-v0 \
    --name "XFormer-PIG-Pretrain" \
    --root-dir ~/cs224r/nanorl/runs/ \
    --warmstart-steps 0 \
    --checkpoint_interval 1000 \
    --max-steps 100000 \
    --discount 0.99 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 40 \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --num_workers 1 \
    --update_period 10 \
    --agent_config.use_transformer \
    --agent_config.num_qs 2 \
    --replay_capacity 1000000 \
    --checkpoint_interval 1000 \
    --log_interval 100 \
    --batch_size 256 \
    --agent_config.actor_lr 3e-4 \
    --agent_config.critic_lr 3e-4 \
    --agent_config.temp_lr 3e-4 \
    --eval_episodes 1 \
    "$@"

    # --agent-config.actor_utd_ratio 14 \
    # --agent-config.critic_utd_ratio 14 \
    # --init_from_checkpoint ~/cs224r/nanorl/runs/SAC-XFormer-PIG-Pretrain-42-1685926327.522851/checkpoint_1100 \