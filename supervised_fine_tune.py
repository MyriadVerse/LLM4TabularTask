# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
# from llama_attn_replace import replace_llama_attn
# from peft import LoraConfig, get_peft_model
from torch.distributed import barrier



IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),
# }

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input_seg}\n\n### Question:\n{question}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

types = 'royalty.noble_person, business.business_operation, protected_sites.listed_site, music.writer, people.ethnicity, government.government_office_or_title, organization.non_profit_organization, business.brand, tennis.tennis_tournament, cvg.cvg_genre, ice_hockey.hockey_position, sports.sports_team, computer.computer, metropolitan_transit.transit_line, award.award_category, american_football.football_conference, sports.professional_sports_team, soccer.football_world_cup, tv.tv_actor, business.industry, music.composition, people.person, broadcast.tv_channel, cricket.cricket_player, internet.website, tennis.tennis_player, music.media_format, tv.tv_personality, film.actor, film.film_genre, cvg.cvg_developer, business.job_title, chess.chess_player, tv.tv_writer, broadcast.broadcast, soccer.fifa, cvg.cvg_publisher, film.writer, medicine.anatomical_structure, astronomy.celestial_object, cricket.cricket_team, sports.golfer, book.periodical_subject, military.rank, spaceflight.astronaut, medicine.disease, location.province, location.location, amusement_parks.ride, government.general_election, music.musical_scale, music.lyricist, music.artist, location.capital_of_administrative_division, theater.play, meteorology.tropical_cyclone, aviation.airport, basketball.basketball_team, education.school, soccer.football_position, soccer.football_team, cvg.cvg_platform, religion.religious_leader, business.defunct_company, astronomy.asteroid, sports.pro_athlete, sports.school_sports_team, baseball.baseball_league, architecture.structure, sports.tournament_event_competition, sports.multi_event_tournament, music.record_label, travel.accommodation, cricket.cricket_stadium, ice_hockey.hockey_team, award.competition, business.consumer_company, people.family_member, biology.organism_classification, business.product_category, book.magazine, royalty.kingdom, fictional_universe.fictional_character, education.athletics_brand, military.military_unit, american_football.football_coach, broadcast.tv_station, government.governmental_body, boats.ship, visual_art.visual_artist, meteorology.tropical_cyclone_season, sports.sports_league, sports.sports_league_season, soccer.football_team_manager, boats.ship_class, military.military_post, education.educational_institution, sports.sports_championship, film.film, award.award_presenting_organization, soccer.football_award, broadcast.artist, computer.software, broadcast.genre, education.university, time.recurring_event, book.periodical, celebrities.celebrity, location.country, soccer.football_player, book.book, geography.river, medicine.drug_ingredient, transportation.road, olympics.olympic_games, military.military_conflict, chemistry.chemical_element, location.us_state, location.hud_county_place, award.award_ceremony, tv.tv_program_creator, architecture.venue, film.music_contributor, architecture.architectural_structure_owner, basketball.basketball_position, astronomy.constellation, law.court, rail.locomotive_class, book.newspaper, film.director, broadcast.radio_station, tv.tv_series_season, architecture.building, olympics.olympic_event_competition, music.instrument, organization.organization, computer.software_license, government.election, award.award, tv.tv_director, metropolitan_transit.transit_system, tennis.tennis_tournament_champion, cricket.cricket_bowler, aviation.airline, tv.tv_network, music.musical_group, government.politician, music.music_video_director, media_common.media_genre, comic_books.comic_book_character, automotive.company, location.administrative_division, government.political_party, location.australian_local_government_area, theater.theater_actor, music.producer, ice_hockey.hockey_player, royalty.monarch, sports.sports_championship_event, sports.sports_league_draft, food.food, military.military_person, geography.island, location.uk_constituent_country, tv.tv_series_episode, government.u_s_congressperson, amusement_parks.park, book.written_work, geography.body_of_water, tv.tv_genre, aviation.aircraft_owner, interests.collection_category, astronomy.star_system_body, tv.tv_producer, medicine.muscle, baseball.baseball_team, government.us_president, location.citytown, fictional_universe.fictional_organization, biology.organism, tv.tv_program, soccer.football_league_season, sports.boxer, military.armed_force, location.australian_state, basketball.basketball_conference, internet.website_owner, medicine.drug, award.award_discipline, location.in_district, business.consumer_product, broadcast.radio_format, baseball.baseball_position, book.periodical_publisher, government.government_agency, sports.cyclist, time.event, automotive.model, boats.ship_type, finance.currency, government.legislative_session, american_football.football_player, royalty.chivalric_order_member, law.invention, martial_arts.martial_artist, film.film_character, sports.sports_facility, music.group_member, location.region, astronomy.orbital_relationship, basketball.basketball_player, cvg.computer_videogame, law.legal_case, language.human_language, tv.tv_character, education.educational_degree, aviation.aircraft_model, business.customer, geography.mountain, location.us_county, music.album, music.composer, computer.operating_system, religion.religion, organization.membership_organization, sports.sport, location.uk_statistical_location, location.in_state, film.film_distributor, basketball.basketball_coach, medicine.medical_treatment, education.fraternity_sorority, metropolitan_transit.transit_stop, chemistry.chemical_compound, sports.sports_position, music.genre, award.hall_of_fame_inductee, sports.sports_award_type, exhibitions.exhibition_sponsor, film.film_festival_focus, film.production_company, location.jp_prefecture, education.field_of_study, award.recurring_competition, government.election_campaign, sports.sports_award_winner, astronomy.astronomical_discovery, music.performance_role, soccer.football_league, book.author, film.producer, royalty.noble_title, biology.animal, american_football.football_team, baseball.baseball_player'

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Formatting inputs...")
        '''
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        '''

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        
        # print(prompt_input.format_map(list_data_dict[1]))
        # print("****")
        # print(f"{list_data_dict[1]['output']}{DEFAULT_EOS_TOKEN}")
        # import pdb
        # pdb.set_trace()

        sources = [
            # prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            prompt_input.format_map(example) if example.get("input_seg", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        # targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        targets = [f"{example['output']}{DEFAULT_EOS_TOKEN}" for example in list_data_dict]

        # sources = [example["instruction"] for example in list_data_dict]

        # targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    replace_llama_attn(training_args.use_flash_attn, True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if training_args.low_rank_training:
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
