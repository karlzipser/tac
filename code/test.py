## 79                                                                       ##
##############################################################################
##                                                                          ##
print(__file__)
assert 'project_' in __file__
from utilz2 import *
import sys,os
import projutils
moving_average=projutils.moving_average
from ..params.runtime import *
from .dataloader import *
from ..net.code.net import *
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision
torchvision.disable_beta_transforms_warning()
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


class GenDataset2(Dataset):
    def __init__(self, root, transform=None):
        print('\n*** GenDataset __init__()')
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        fs0=sggo(root,'*')
        for cf in fs0:
            if not os.path.isdir(cf):
                continue
            for image in sggo(cf,'*.png'):
                self.images.append(image)
                self.labels.append(fname(cf))
                #print(self.images[-1],self.labels[-1])
        print('\tlen(self.images)=',
            len(self.images),'len(self.labels)=',len(self.labels))
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = rimread(self.images[index])
        if self.transform:
            image = self.transform(image)
        return image,classes[self.labels[index]],self.images[index]



##                                                                          ##
##############################################################################
##                                                                          ##
device = torch.device(p.device if torch.cuda.is_available() else 'cpu')
##                                                                          ##
##############################################################################
##                                                                          ##
if p.run_path:
    cy('\n*** Continuing from p.run_path=',p.run_path,r=1)
    for task in p.data_recorders:
        p.data_recorders[task].load(opjh(p.run_path,fname(thispath),'stats'))
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

##                                                                          ##
##############################################################################
##                                                                          ##
print('*** Start Testing gen data. . .')

task='gen_trainloader'
p.data_recorders[task].dataloader=DataLoader(
    GenDataset2(
        root=p.gen_data_path,
        transform=test_transform),
    batch_size=p.batch_size,
    shuffle=False)
p.data_recorders[task].dataiter=iter(p.data_recorders[task].dataloader)
assert p.batch_size==1

net.eval()
h=[]
for ig in range(10**20):
    
    if True:#try:
            
        printr(d2n(thispath.replace(opjh(),''),': ig=',ig,', t=',int(p.timer.max.time())))

        inputs,labels,files=next(p.data_recorders[task].dataiter)

        inputs=inputs.to(device)
        #
        ##########################################################################
        #
        outputs = net(inputs)
        outputs=outputs.detach()
        outputs[outputs<0]=0
        targets=0*outputs.detach()
        for i in range(targets.size()[0]):
            targets[i,labels[i],0,0]=1
        loss = p.criterion(torch.flatten(outputs,1),torch.flatten(targets,1))

        #print(outputs.size(),labels.size())
        a=outputs[0,labels.item(),0,0].cpu().numpy()
        b=outputs.sum().item().cpu().numpy()
        c=a/b
        if np.isnan(c):
            c=0
        h.append(c)
        #hist(h)
        if c>=0.5:
            figure(1)
            projutils.show_sample_outputs(
                inputs,
                outputs,
                labels,
                ig,
                d2s(int(100*c),len(h)),
                '',
                )
            spause()


        continue
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
                projutils.show_sample_outputs(
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
            save_path=opj(paths.figures)

            projutils.reduce_file_numbers(save_path,p.max_num_samples)
            
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
            plt.title('loss')
            plt.legend(kys(p.data_recorders),loc='upper right')
            plt.savefig(
                opj(save_path,get_safe_name('loss')+'.pdf'),
                bbox_inches='tight')

    try:
        pass
    except KeyboardInterrupt:
        cr('*** KeyboardInterrupt ***')
        sys.exit()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        cE('Exception!')
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