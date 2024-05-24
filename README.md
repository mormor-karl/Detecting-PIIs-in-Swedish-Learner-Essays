# Detecting-PIIs-in-Swedish-Learner-Essays

This is the repository for the [Detecting Personal Identifiable Information in Swedish Learner Essays (Szawerna et al., CALD-pseudo-WS 2024)](https://aclanthology.org/2024.caldpseudo-1.7/) paper, in which we investigate the possibility of a) using Swedish BERT for detecting PIIs in L2 learner essays and b) using a simple IOB annotation to signify the PII vs. not PII difference. Out of respect for the privacy of the data subjects and legal concerns we are unable to share the original data. One can apply for the access to the already pseudonymized SweLL data [here](https://sunet.artologik.net/gu/swell). 

### Preparations
The token classification with Transformers is based off of [this example](https://github.com/huggingface/transformers/tree/main/examples/legacy/token-classification). Please make sure that you have this code saved. In our case we had it in a subfolder in this repository called `bert` (which in this repository only contains our custom file for running everything). Once you have the code, the following steps need to be carried out in order to enable the weighted loss function option:

1. Locate the `run_ner.py` file.
2. Replace lines 247 to 255 (initializing a Trainer() object) with the following:
```
    weighted = True  # change to False for not weighted
    # custom trainer to be used with weighted CrossEntropyLoss
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 3 labels with different weights)
            if weighted:  
                loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([12.64419148, 167.90310078, 0.34305829], device=model.device))  
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
```
3. The weights in the weighted can be altered if needed. Unfortunately, the switching between weighted and not weighted has to be done in the file.

### Re-running the experiments
Once this preparation is done, and if you have the appropriate SweLL-pilot files, you can do the following to re-run the experiments:
1. In the main folder run `python3 reannotate_iob.py [INPUT/SWELL FOLDER] [OUTPUT FOLDER] [optional flags]`
2. `cd ./data/`
3. `sh preproc.sh` (optionally also `cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt`, needed only once if you don't have the labels file)
4. `cd ../bert/`
5. `sh run_iob.sh` (note: if you want to toggle between weighted or unweighted CrossEntropyLoss, you have to do it manually in `run_ner.py`; same goes for changing/adding settings in `run_iob.sh`)
6. `cd ../`
7. `python3 analyze_output.py [OUTPUT FOLDER] [MODEL NAME]` for each of the custom trained models.

## License
This code is released under the [CRAPL academic-strength open source license](https://matt.might.net/articles/crapl/).
