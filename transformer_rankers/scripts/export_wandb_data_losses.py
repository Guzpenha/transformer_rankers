import pandas as pd 
import wandb
from IPython import embed

api = wandb.Api()
entity, project = "guz", "table_diff_loss"  # set to your entity and project 
runs = api.runs(entity + "/" + project) 

plots = []
summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_dict = {k: v for k,v in run.config.items()
         if not k.startswith('_')}
    config_list.append(config_dict)

    # .name is the human-readable name of the run.
    name_list.append(run.name)
    df_plot = run.history()[["_step", "MAP"]]
    df_plot['task'] = config_dict['task']
    df_plot['loss'] = config_dict['loss']
    df_plot['NegativeSampler'] = config_dict['negative_sampler']
    df_plot = df_plot.replace(
        {'mantis':'MANtIS', 
        'msdialog':'MSDialog',
        'ubuntu_dstc8':'UDC',
        'bm25': "BM25 (1a)",
        'random': "Random (0)",
        'sentence_transformer':"Bi-encoder (3e)",
        }
        )
    plots.append(df_plot)

df_plots = pd.concat(plots)
df_plots.to_csv("diff_loss_plot.csv", index=False)
runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

df_parsed = []
for idx, values in runs_df.iterrows():    
    if 'R@10' in values['summary']:
        df_parsed.append([
            values['config']['task'],
            values['config']['loss'],
            values['config']['negative_sampler'],
            values['summary']['R@1'],
            values['summary']['R@10']
            ])
df_parsed = pd.DataFrame(df_parsed, columns = ['task', 'loss', 'negative_sampler', 'R@1', 'R@10'])
to_print_df = (df_parsed.sort_values(['task', 'loss', 'R@10']).
        pivot(index=['loss', 'negative_sampler'] , values=['R@10','R@1'], columns = ['task']).reset_index()
            [[(             'loss',             ''),            
            ( 'negative_sampler',             ''),
            (              'R@1',       'mantis'),
            (             'R@10',       'mantis'),
            (              'R@1',     'msdialog'),
            (             'R@10',     'msdialog'),
            (              'R@1', 'ubuntu_dstc8'),
            (             'R@10', 'ubuntu_dstc8')]])
print(to_print_df)
to_print_df.to_csv("diff_loss.csv", index=False, sep='\t')
embed()

            

# runs_df.to_csv("project.csv")