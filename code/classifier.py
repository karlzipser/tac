## 79                                                                       ##
##############################################################################
##                                                                          ##
print(__file__)
assert 'project_' in __file__
from utilz2 import *
import sys,os
import projutils
from ..params.runtime import *
from .dataloader import trainloader,testloader,classes
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

data_recorders=dict(
    train=projutils.net_data_recorder.Data_Recorder(
        dataloader=trainloader,
        name='train',
        ),
    test=projutils.net_data_recorder.Data_Recorder(
        dataloader=testloader,
        name='test',
        ),
    )

if p.run_path:
    print('\n*** Continuing from p.run_path=',p.run_path)
    for task in data_recorders:
        data_recorders[task].load(opjh(p.run_path,fname(thispath),'stats'))
        #cb('loaded',task,len(data_recorders[task].processed))
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

task_list=['train']*5+['test']*1

for ig in range(10**20):
    
    if p.timer.max.check():
        break
    if p.timer.epoch.rcheck():
        break

    if printr_timer.rcheck():
        l=[]
        for task in data_recorders:
            l.append(len(data_recorders[task].processed))
        l=tuple(l)
        printr(d2n(thispath.replace(opjh(),''),': ig=',ig,', t=',int(p.timer.max.time()),'s processed=',l))

    task=np.random.choice(task_list)

    try:
        inputs,labels=next(data_recorders[task].dataiter)
    except:
        data_recorders[task].dataiter=iter(data_recorders[task].dataloader)
        inputs,labels=next(data_recorders[task].dataiter)
    inputs=inputs.to(device)

    #
    ##########################################################################
    # 
    if 'test' not in data_recorders[task].name:
        assert 'train' in data_recorders[task].name
        net.train()
        if p.noise_level and rnd()<p.noise_p:
            inputs+=rnd()*p.noise_level*torch.randn(
                inputs.size()).to(device)
        optimizer.zero_grad()
    else:
        net.eval()
    
    outputs = net(inputs)
    targets=0*outputs.detach()
    for i in range(targets.size()[0]):
        targets[i,labels[i],0,0]=1
    
    loss = p.criterion(torch.flatten(outputs,1),torch.flatten(targets,1))
    if 'test' not in data_recorders[task].name:
        loss.backward()
        optimizer.step()
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

    data_recorders[task].add(_d)

    #
    ##########################################################################
    #    

    if p.timer.show.rcheck():
        for task in data_recorders:
            #cb(task)
            latest=data_recorders[task].latest()
            if not latest:
                continue
            show_sample_outputs(
                latest['inputs'],
                latest['outputs'],
                latest['labels'],
                ig,
                data_recorders[task].name+'_outputs',
                save_path=opj(paths.figures),
                )
            sh(torchvision.utils.make_grid(latest['inputs']),
                title=data_recorders[task].name+'_examples',use_spause=False,
                    save_path=opj(paths.figures))
        #
        ######################################################################
        #    
        from sklearn.metrics import f1_score
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        save_path=opj(paths.figures)

        if True:#try:

            for task in data_recorders:

                processed=data_recorders[task].processed
                n=max(1,len(processed)//100)
                if not len(processed):
                    continue
                figure(1);clf()
                for c in classes:
                    f=[]
                    igs=[]
                    for pr in processed:
                        f.append(pr['accuracy'][classes[c]])
                        igs.append(pr['ig'])
                    x=moving_average(igs,n)
                    y=moving_average(f,n)
                    plot(x,y,label=classes[c])
                plt.title(data_recorders[task].name+' accuracy')
                plt.legend(kys(classes),loc='upper left')
                plt.savefig(
                    opj(save_path,
                        data_recorders[task].name+'-'+get_safe_name(
                            'accuracy')+'.pdf'),
                    bbox_inches='tight')


                figure(1);clf()
                for c in classes:
                    f=[]
                    igs=[]
                    for pr in processed:
                        f.append(pr['f1_scores'][classes[c]])
                        igs.append(pr['ig'])
                    x=moving_average(igs,n)
                    y=moving_average(f,n)
                    plot(x,y,label=classes[c])
                plt.title(data_recorders[task].name+' f1-scores')
                plt.legend(kys(classes),loc='upper left')
                plt.savefig(
                    opj(save_path,data_recorders[
                        task].name+'-'+get_safe_name(
                            'f1-scores')+'.pdf'),
                    bbox_inches='tight')


                fig=figure(1);clf()
                ax=fig.add_subplot(111)
                a=max(1,int(len(processed)*0.9))
                b=len(processed)
                c=[]
                for i in range(a,b):
                    c.append(processed[i]['confusion_matrix'])
                c=na(c)
                c=c.sum(axis=0)
                c=(100*c.astype(
                    'float')/c.sum(axis=1)[:,np.newaxis]).astype(int)
                if False:
                    print(c)
                    print(c.sum(axis=1))
                disp=ConfusionMatrixDisplay(
                    confusion_matrix=c,
                    display_labels=kys(classes))
                disp.plot(ax=ax,cmap=plt.cm.Blues)
                plt.title(
                    d2s(data_recorders[task].name,'confusion_matrix',ig))
                plt.savefig(
                    opj(save_path,
                        data_recorders[task].name+'-'+get_safe_name(
                            'confusion_matrix.pdf')),
                    bbox_inches='tight')

            figure(1)
            clf()
            for task in data_recorders:
                processed=data_recorders[task].processed
                f=[]
                igs=[]
                for pr in processed:
                    f.append(pr['loss'])
                    igs.append(pr['ig'])
                x=moving_average(igs,n)
                y=moving_average(f,n)
                plot(x,y,label=data_recorders[task].name)
            plt.title('loss')
            plt.legend(kys(data_recorders),loc='upper right')
            plt.savefig(
                opj(save_path,get_safe_name('loss')+'.pdf'),
                bbox_inches='tight')
        """
        except KeyboardInterrupt:
            cr('*** KeyboardInterrupt ***')
            sys.exit()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('Exception!')
            print(d2s(exc_type,file_name,exc_tb.tb_lineno)) 
        """

    #
    ##########################################################################
    #    



    if p.timer.save.rcheck():
        projutils.net_access.save_net(net,paths.weights_latest)
        for task in data_recorders:
            data_recorders[task].save(paths.stats)

print('*** Finished Training')


##                                                                          ##
##############################################################################
## EOF                                                                      ##