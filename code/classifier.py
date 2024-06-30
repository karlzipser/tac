## 79                                                                       ##
##############################################################################
##                                                                          ##
print(__file__)
assert 'project_' in __file__
from utilz2 import *
import sys,os
import projutils
from ..params.runtime import *
from .dataloader import loader_dic,classes
from ..net.code.net import *
##                                                                          ##
##############################################################################
##                                                                          ##
thispath=pname(pname(__file__))
sys.path.insert(0,opj(thispath,'env'))
paths=k2c()#ti='paths')
paths.weights=opj(thispath,'net/weights')
paths.figures=opj(thispath,'figures')
paths.stats=opj(thispath,'stats')
mkdirp(paths.figures)
mkdirp(paths.weights)
mkdirp(paths.stats)
paths.weights_latest=opj(paths.weights,'latest.pth')
paths.weights_best=  opj(paths.weights,'best.pth')
##                                                                          ##
##############################################################################
##                                                                          ##
device = torch.device(p.device if torch.cuda.is_available() else 'cpu')
##                                                                          ##
##############################################################################
##                                                                          ##
def show_sample_outputs(inputs,outputs,labels,ig,name,save_path):
    outputs=outputs.detach().cpu().numpy()
    labels=labels.detach().cpu().numpy()
    if False:
        print(outputs)
        print(labels)
        print(shape(outputs))
        print(shape(labels))
    i=0
    o=outputs[i,:,0,0]
    l=0*o
    l[labels[i]]=1
    sh(cuda_to_rgb_image(inputs[i,:]),use_spause=False)
    xs=np.arange(len(o))/len(o)*inputs.size()[2]
    plot(xs,inputs.size()[2]-o/o.max()*inputs.size()[2],'r')
    plot(xs,inputs.size()[2]-l*inputs.size()[2],'b')
    title(d2s(name,ig))
    if save_path:
        plt.savefig(
            opj(save_path,str(ig)+'-'+get_safe_name(name)+'.png'),
            bbox_inches='tight')
##                                                                          ##
##############################################################################
##                                                                          ##




if p.run_path:
    cg('\n*** Continuing from p.run_path=',p.run_path,r=1)
    for task in p.data_recorders:
        p.data_recorders[task].load(opjh(p.run_path,fname(thispath),'stats'))
        #cb('loaded',task,len(p.data_recorders[task].processed))
    net=projutils.net_access.get_net(
        device=device,
        run_path=p.run_path,
        latest=True,
    )
else:
    net=projutils.net_access.get_net(device=device,net_class=Net)
##                                                                          ##
##############################################################################
##                                                                          ##
if p.opt==optim.Adam:
    optimizer = p.opt(net.parameters(),lr=p.lr)
else:
    optimizer = p.opt(net.parameters(),lr=p.lr,momentum=p.momentum)
p.timer.show.trigger()
p.timer.save.trigger()
p.timer.epoch.trigger()
best_loss=1e999
printr_timer=Timer(1)


def moving_average(data, window_size):
    """ Smooth data using a moving average """
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

##                                                                          ##
##############################################################################
##                                                                          ##
print('*** Start Training . . .')



for ig in range(10**20):
    try:
        if p.timer.max.check():
            break
        if p.timer.epoch.rcheck():
            print('\n*** epoch '+50*'*','\n')
            for k in p.data_recorders:
                p.data_recorders[k].dataloader=loader_dic[p.data_recorders[k].dataloader]


        if printr_timer.rcheck():
            l=[]
            for task in p.data_recorders:
                l.append(len(p.data_recorders[task].processed))
            l=tuple(l)
            printr(d2n(thispath.replace(opjh(),''),': ig=',ig,', t=',int(p.timer.max.time()),'s processed=',l))

        task=np.random.choice(p.task_list)

        try:
            inputs,labels=next(p.data_recorders[task].dataiter)
        except:
            p.data_recorders[task].dataiter=iter(p.data_recorders[task].dataloader)
            inputs,labels=next(p.data_recorders[task].dataiter)
        inputs=inputs.to(device)


        #
        ##########################################################################
        #
        if p.data_recorders[task].noise_level and rnd()<p.data_recorders[task].noise_p:
            inputs+=rnd()*p.data_recorders[task].noise_level*torch.randn(
                inputs.size()).to(device)

        if 'test' not in p.data_recorders[task].name:
            assert 'train' in p.data_recorders[task].name
            net.train()
            optimizer.zero_grad()
        else:
            net.eval()
        
        outputs = net(inputs)
        targets=0*outputs.detach()
        if not p.data_recorders[task].targets_to_zero:
            for i in range(targets.size()[0]):
                targets[i,labels[i],0,0]=1
        
        loss = p.criterion(torch.flatten(outputs,1),torch.flatten(targets,1))
        if 'test' not in p.data_recorders[task].name:
            loss.backward()
            optimizer.step()

        #cE(inputs.size(),outputs.size(),targets.size())
        #
        ##########################################################################
        # 
        _d=dict(
                ig=ig,
                inputs=inputs.detach().cpu(),
                labels=labels.detach().cpu(),
                outputs=outputs.detach().cpu(),
                targets=targets.detach().cpu(),
                loss=loss.detach().cpu()
            )

        p.data_recorders[task].add(_d)

        #
        ##########################################################################
        #    

        if p.timer.show.rcheck():
            for task in p.data_recorders:
                #cb(task)
                latest=p.data_recorders[task].latest()
                if not latest:
                    continue
                show_sample_outputs(
                    latest['inputs'],
                    latest['outputs'],
                    latest['labels'],
                    ig,
                    p.data_recorders[task].name+'_outputs '+kys(classes)[int(latest['labels'][0].item())],
                    save_path=opj(paths.figures),
                    )
                sh(torchvision.utils.make_grid(latest['inputs']),
                    title=p.data_recorders[task].name+'_examples',use_spause=False,
                        save_path=opj(paths.figures))
            #
            ######################################################################
            #    
            from sklearn.metrics import f1_score
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            save_path=opj(paths.figures)

            if True:#try:

                for task in p.data_recorders:

                    processed=p.data_recorders[task].processed
                    n=max(1,len(processed)//100)
                    if not len(processed):
                        continue
                    fig=figure(1);clf();ax=fig.add_subplot(111)
                    for c in classes:
                        f=[]
                        igs=[]
                        for pr in processed:
                            f.append(pr['accuracy'][classes[c]])
                            igs.append(pr['ig'])
                        x=moving_average(igs,n)
                        y=moving_average(f,n)
                        plot(x,y,label=classes[c])
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                    ax.grid(which='major', linestyle=':', linewidth='0.5', color='black')
                    ax.minorticks_on()
                    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='grey', alpha=0.7)
                    plt.title(p.data_recorders[task].name+' accuracy')
                    plt.legend(kys(classes),loc='upper left')
                    plt.savefig(
                        opj(save_path,
                            p.data_recorders[task].name+'-'+get_safe_name(
                                'accuracy')+'.pdf'),
                        bbox_inches='tight')


                    fig=figure(1);clf();ax=fig.add_subplot(111)
                    for c in classes:
                        f=[]
                        igs=[]
                        for pr in processed:
                            f.append(pr['f1_scores'][classes[c]])
                            igs.append(pr['ig'])
                        x=moving_average(igs,n)
                        y=moving_average(f,n)
                        plot(x,y,label=classes[c])
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                    ax.grid(which='major', linestyle=':', linewidth='0.5', color='black')
                    ax.minorticks_on()
                    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='grey', alpha=0.7)
                    plt.title(p.data_recorders[task].name+' f1-scores')
                    plt.legend(kys(classes),loc='upper left')
                    plt.savefig(
                        opj(save_path,p.data_recorders[
                            task].name+'-'+get_safe_name(
                                'f1-scores')+'.pdf'),
                        bbox_inches='tight')


                    try:
                        if len(processed)>1:
                            fig=figure(1);clf()
                            ax=fig.add_subplot(111)
                            a=max(1,int(len(processed)*0.9))
                            b=len(processed)
                            c=[]
                            for i in range(a,b):
                                c.append(processed[i]['confusion_matrix'])
                            c=na(c)
                            c=c.sum(axis=0)
                            c=np.round((100*c.astype(
                                'float')/c.sum(axis=1)[:,np.newaxis])).astype(int)

                            disp=ConfusionMatrixDisplay(
                                confusion_matrix=c,
                                display_labels=kys(classes))
                            disp.plot(ax=ax,cmap=plt.cm.Blues)
                            plt.title(
                                d2s(p.data_recorders[task].name,'confusion_matrix',ig))
                            plt.savefig(
                                opj(save_path,
                                    p.data_recorders[task].name+'-'+get_safe_name(
                                        'confusion_matrix.pdf')),
                                bbox_inches='tight')
                    except KeyboardInterrupt:
                        cr('*** KeyboardInterrupt ***')
                        sys.exit()
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print('Exception!')
                        print(d2s(exc_type,file_name,exc_tb.tb_lineno)) 



                fig=figure(1);clf();ax=fig.add_subplot(111)
                for task in p.data_recorders:
                    processed=p.data_recorders[task].processed
                    f=[]
                    igs=[]
                    for pr in processed:
                        f.append(pr['loss'])
                        igs.append(pr['ig'])
                    x=moving_average(igs,n)
                    y=moving_average(f,n)
                    plot(x,y,label=p.data_recorders[task].name)

                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.grid(which='major', linestyle=':', linewidth='0.5', color='black')
                ax.minorticks_on()
                ax.grid(which='minor', linestyle=':', linewidth='0.5', color='grey', alpha=0.7)


                #ax.yaxis.grid(True, which='both', linestyle=':', linewidth='0.5', color='black')
                #ax.xaxis.grid(False)

                plt.title('loss')
                plt.legend(kys(p.data_recorders),loc='upper right')
                plt.savefig(
                    opj(save_path,get_safe_name('loss')+'.pdf'),
                    bbox_inches='tight')
    
    except KeyboardInterrupt:
        cr('*** KeyboardInterrupt ***')
        sys.exit()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        CE('Exception!')
        print(d2s(exc_type,file_name,exc_tb.tb_lineno)) 
    

    #
    ##########################################################################
    #    



    if p.timer.save.rcheck():
        projutils.net_access.save_net(net,paths.weights_latest)
        for task in p.data_recorders:
            p.data_recorders[task].save(paths.stats)

print('*** Finished Training')


##                                                                          ##
##############################################################################
## EOF                                                                      ##