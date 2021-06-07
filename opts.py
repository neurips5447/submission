import argparse,os,re
import configparser
import sys

def parse_opt():

    parser = argparse.ArgumentParser()
    # Data input settings

    parser.add_argument('--if_test', type=str, default="true", help='')

    parser.add_argument('--torch_seed', type=float, default=0, help='')
    parser.add_argument('--rand_seed', type=float, default=1234, help='')

    parser.add_argument('--imdb_epochs', type=int, default=21, help='')

    parser.add_argument('--bert_type', type=str, default="bert", help='')#roberta

    parser.add_argument('--ascc_mode', type=str, default='comb_p', help='')

    parser.add_argument('--work_path', type=str, default='./', help='')
    
    parser.add_argument('--h_test_start', type=int, default=0, help='')

    parser.add_argument('--snli_epochs', type=int, default=11, help='')

    parser.add_argument('--lm_constraint', type=str, default='true', help='')

    parser.add_argument('--imdb_lm_file_path', type=str, default="lm_scores/imdb_all.txt", help='') 
    parser.add_argument('--snli_lm_file_path', type=str, default="lm_scores/snli_all_save.txt", help='') 

    parser.add_argument('--certified_neighbors_file_path', type=str, default="counterfitted_neighbors.json", help='') 

    parser.add_argument('--train_attack_sparse_weight', type=float, default=15, help='') 

    parser.add_argument('--attack_sparse_weight', type=float, default=15, help='') 

    parser.add_argument('--w_optm_lr', type=float, default=10, help='') 

    parser.add_argument('--bert_w_optm_lr', type=float, default=1, help='') 

    parser.add_argument('--freeze_bert_stu', type=str, default='false', help='') 
    parser.add_argument('--freeze_bert_tea', type=str, default='true', help='') 
    
    parser.add_argument('--resume', type=str, default=None, help='') 

    parser.add_argument('--pwws_test_num', type=int, default=500, help='') 
    parser.add_argument('--genetic_test_num', type=int, default=500, help='') 
    parser.add_argument('--genetic_iters', type=int, default=40, help='') 
    parser.add_argument('--genetic_pop_size', type=int, default=60, help='') 


    parser.add_argument('--weight_adv', type=float, default=0, help='') 
    parser.add_argument('--weight_clean', type=float, default=0, help='') 
    parser.add_argument('--weight_ball', type=float, default=0, help='') 
    parser.add_argument('--weight_kl', type=float, default=0, help='') 
    
    parser.add_argument('--weight_mi_clean', type=float, default=0, help='') 
    parser.add_argument('--weight_mi_adv', type=float, default=0, help='') 
    parser.add_argument('--weight_mi_giveny_clean', type=float, default=0, help='') 
    parser.add_argument('--weight_mi_giveny_adv', type=float, default=0, help='') 
    parser.add_argument('--weight_params_l2', type=float, default=0, help='') 

    parser.add_argument('--infonce_sim_metric', type=str, default='projected_cossim', help='') 

    parser.add_argument('--test_attack_iters', type=int, default=10,
                    help='') 
    parser.add_argument('--test_attack_eps', type=float, default=1,
                    help='') 
    parser.add_argument('--test_attack_step_size', type=float, default=0.25,
                    help='') 

    parser.add_argument('--random_start', type=str, default='true', help='')

    parser.add_argument('--train_attack_iters', type=int, default=10,
                    help='') 
                    
    parser.add_argument('--train_attack_eps', type=float, default=5.0, help='') 

    parser.add_argument('--train_attack_step_size', type=float, default=0.25,
                    help='') 

    parser.add_argument('--imdb_synonyms_file_path', type=str, default="temp/imdb.synonyms",
                    help='')

    parser.add_argument('--imdb_bert_synonyms_file_path', type=str, default="temp/imdb.bert.synonyms",
                    help='')

    parser.add_argument('--imdb_roberta_synonyms_file_path', type=str, default="temp/imdb.roberta.synonyms",
                    help='')
    
    parser.add_argument('--imdb_xlnet_synonyms_file_path', type=str, default="temp/imdb.xlnet.synonyms",
                help='')

    parser.add_argument('--snli_synonyms_file_path', type=str, default="temp/snli.synonyms",
                    help='')

    parser.add_argument('--snli_bert_synonyms_file_path', type=str, default="temp/snli.bert.synonyms",
                    help='')
    
    parser.add_argument('--snli_roberta_synonyms_file_path', type=str, default="temp/snli.roberta.synonyms",
                help='')
    parser.add_argument('--snli_xlnet_synonyms_file_path', type=str, default="temp/snli.xlnet.synonyms",
                help='')


    parser.add_argument('--synonyms_from_file', type=str, default='true',
                    help='')

    parser.add_argument('--out_path', type=str, default="./",
                    help='')

    parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size') 

    parser.add_argument('--test_batch_size', type=int, default=32,
                    help='test_batch_size') 
    
    parser.add_argument('--learning_rate', type=float, default=0.005, help='learning_rate')

    parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight_decay')

    parser.add_argument('--optimizer', type=str, default="adamw", help='optimizer')

    parser.add_argument('--lr_scheduler', type=str, default="none", help='lr_scheduler')

    parser.add_argument('--grad_clip', type=float, default=1e-1, help='grad_clip')

    parser.add_argument('--model', type=str, default="bert_adv_kd",
                    help='model name')

    parser.add_argument('--dataset', type=str, default="imdb",
                    help='dataset')

    parser.add_argument('--embedding_file_path', type=str, default="glove/glove.840B.300d.txt",
                    help='glove or w2v')

    parser.add_argument('--embedding_training', type=str, default="false",
                    help='embedding_training')

    parser.add_argument('--gpu', type=int, default=0,
                    help='gpu number')

    parser.add_argument('--gpu_num', type=int, default=1,
                    help='gpu number')

    parser.add_argument('--proxy', type=str, default="null",
                    help='http://proxy.xx.com:8080')

    parser.add_argument('--debug', type=bool, default=False,
                    help='')
    
#
    args = parser.parse_args()


    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] =str(args.gpu)
    
    # process the type for bool and list    
    for arg in args.__dict__.keys():
        if type(args.__dict__[arg])==str:
            if args.__dict__[arg].lower()=="true":
                args.__dict__[arg]=True
            elif args.__dict__[arg].lower()=="false":
                args.__dict__[arg]=False
            elif "," in args.__dict__[arg]:
                args.__dict__[arg]= [int(i) for i in args.__dict__[arg].split(",") if i!='']
            else:
                pass

    sys.path.append(args.work_path)

    for arg in args.__dict__.keys():
        if "path" in arg and arg!="work_path":
            args.__dict__[arg] = os.path.join(args.work_path, args.__dict__[arg])

    if os.path.exists("proxy.config"):
        with open("proxy.config") as f:
            args.proxy = f.read()
            print(args.proxy)
    
    return args 