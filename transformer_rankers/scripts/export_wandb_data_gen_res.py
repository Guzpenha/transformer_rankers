import pandas as pd 
import wandb
from IPython import embed

api = wandb.Api()
# entity, project = "guz", "context_part_of_input"  # set to your entity and project 
# entity, project = "guz", "generated_negative_samples"  # set to your entity and pro
# entity, project = "guz", "last_utterance_only"  # set to your entity and project ject 
entity, project = "guz", "expanded_candidates"  # set to your entity and project ject 
runs = api.runs(entity + "/" + project) 

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })
# embed()
df_parsed = []
for idx, values in runs_df.iterrows():    
    if 'R@10' in values['summary']:
        df_parsed.append([
            values['config']['task'],
            values['config']['transformer_model'],
            values['config']['negative_sampler'],
            values['config']['generative_sampling_model'],
            values['config']['loss'],
            values['summary']['R@1'],
            values['summary']['R@10']
            ])
# embed()            
df_parsed = pd.DataFrame(df_parsed, columns = ['task', 'transformer_model', 'negative_sampler', 'generative_sampling_model','loss', 'R@1', 'R@10'])
to_print_df = (df_parsed[df_parsed["loss"] == "MultipleNegativesRankingLoss"].sort_values(['task', 'transformer_model', 'R@10']).
        # pivot(index=['loss', 'transformer_model', 'generative_sampling_model'] , values=['R@10','R@1'], columns = ['task']).reset_index()
        pivot(index=['negative_sampler'] , values=['R@10','R@1'], columns = ['task']).reset_index()            
            [[
                # (             'loss',             ''),
            # ('transformer_model',             ''),
            # ('generative_sampling_model',             ''),
            ('negative_sampler',             ''),
            (              'R@1',       'mantis'),
            (             'R@10',       'mantis'),
            (              'R@1',     'msdialog'),
            (             'R@10',     'msdialog'),
            (              'R@1', 'ubuntu_dstc8'),
            (             'R@10', 'ubuntu_dstc8')]])
print(to_print_df)
# to_print_df.to_csv("diff_gen_models.csv", index=False, sep='\t')
# to_print_df.to_csv("context_part_input.csv", index=False, sep='\t')
# to_print_df.to_csv("last_utterance_sampling.csv", index=False, sep='\t')
to_print_df.to_csv("expanded_candidates.csv", index=False, sep='\t')
embed()

            

# runs_df.to_csv("project.csv")