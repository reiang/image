import datetime
import matplotlib.pyplot as plt
import numpy as np
import pywt
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from critic import BasicCritic
from decoder import DenseDecoder
from encoder import DenseEncoder, BasicEncoder, ResidualEncoder
from torchvision import datasets, transforms
from IPython.display import clear_output
import torchvision
from torch.optim import Adam
import pytorch_ssim
from tqdm import tqdm
import torch
import os
import gc
from PIL import ImageFile
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import pdb
# RL
from agent.DQN_agent import DQN
from agent.RL_env import compute_reward, formulate_state
from agent.Fixed_agent import FixedAgent

from RS_analysis import rs_analysis

ImageFile.LOAD_TRUNCATED_IMAGES = True



# plot('encoder_mse', ep, metrics['val.encoder_mse'], True)
def plot(name, train_epoch, values, save):
    clear_output(wait=True)
    plt.close('all')
    fig = plt.figure()
    fig = plt.ion()
    fig = plt.subplot(1, 1, 1)
    fig = plt.title('epoch: %s -> %s: %s' % (train_epoch, name, values[-1]))
    fig = plt.ylabel(name)
    fig = plt.xlabel('train_loader')  # epoch??
    fig = plt.plot(values)
    fig = plt.grid()
    get_fig = plt.gcf()
    fig = plt.draw()  # draw the plot
    # fig = plt.pause(1)  # show it for 1 second
    if save:
        # now = datetime.datetime.now()
        get_fig.savefig('results/plots/%s_%d.png' %
                        (name, train_epoch))
        
def uiqi(reference_image, distorted_image):
    # 将图像转换为灰度
    transform_to_gray = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # 使用ToPILImage转换
    transform_to_pil_image = transforms.ToPILImage()

    # 将张量转换为PIL图像
    ref_PIL = transform_to_pil_image(reference_image)
    dist_PIL = transform_to_pil_image(distorted_image)
    
    ref_gray = transform_to_gray(ref_PIL).unsqueeze(0)
    dist_gray = transform_to_gray(dist_PIL).unsqueeze(0)
    
    # 计算平均亮度
    mu1 = ref_gray.mean()
    mu2 = dist_gray.mean()
    
    # 计算协方差
    sigma_12 = ((ref_gray - mu1) * (dist_gray - mu2)).mean()
    
    # UIQI公式中的常数alpha
    alpha = 1e-10
    
    # 计算UIQI
    uiqi_value = (4 * mu1 * mu2 + 2 * sigma_12 + alpha) / (4 * mu1.pow(2) + 4 * mu2.pow(2) + alpha)
    
    return uiqi_value


def main():
    data_dir = 'div2k'
    epochs = 300
    max_data_depth = 8
    hidden_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
        print('run on GPU')
    else:
        print('run on CPU')
    
    

    METRIC_FIELDS = [
        'val.encoder_mse',
        'val.decoder_loss',
        'val.decoder_acc',
        'val.cover_score',
        'val.generated_score',
        'val.ssim',
        'val.psnr',
        'val.bpp',

        'train.message_density',
        'train.UIQI',
        'train.rs',

        'train.encoder_mse',
        'train.decoder_loss',
        'train.decoder_acc',
        'train.cover_score',
        'train.generated_score',
    ]

    mu = [.5, .5, .5]
    sigma = [.5, .5, .5]
    # 数据预处理
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),  # 以一定的概率随机水平翻转图像
                                    transforms.RandomCrop(
                                        360, pad_if_needed=True),  # 会随机裁剪图像。它将图像裁剪为指定的大小（360x360），如果图像的尺寸小于指定大小，则会进行填充。
                                    transforms.ToTensor(),  # 将图像数据转换为张量（tensor）格式
                                    transforms.Normalize(mu,
                                                         sigma)])  # 这个操作对图像进行标准化处理，将图像的像素值减去均值(mu)并除以标准差(sigma)。这样做可以使得图像的每个通道具有零均值和单位方差，有助于模型的训练。
    
    # 训练集
    train_set = datasets.ImageFolder(os.path.join(  # 加载数据集，并进行预处理
        data_dir, "train/"), transform=transform)
    # train_loader = torch.utils.data.DataLoader(  # 将数据集封装成可迭代的数据加载器，样本数为4，shuffle表示是否对数据进行洗牌
    #     train_set, batch_size=4, shuffle=True)
    # 验证集
    valid_set = datasets.ImageFolder(os.path.join(
        data_dir, "val/"), transform=transform)
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_set, batch_size=4, shuffle=False)

    algs = ['DQN', 'Basic-8',  'Residual-8', 'Dense-8',]
    # algs = ['DQN', 'Residual-4',]

    h1_reward_alg = {alg: [] for alg in algs}
    h0_reward_alg = {alg: [] for alg in algs}
    reward_alg = {alg: [] for alg in algs}
    psnr_alg = {alg: [] for alg in algs}
    ssim_alg = {alg: [] for alg in algs}
    consumption_alg = {alg: [] for alg in algs}
    uiqi_alg = {alg: [] for alg in algs}
    rs_alg = {alg: [] for alg in algs}
    mse_alg = {alg: [] for alg in algs}

    for alg in tqdm(algs):
    
        h1_reward_avg = []
        h0_reward_avg = []
        reward_avg = []
        psnr_avg = []
        ssim_avg = []
        consumption_avg = []
        uiqi_avg = []
        rs_avg = []
        mse_avg = []

        action_log = {
            'mode': [],
            'depth': [],
            'Combination': [],
        }

        avg_times = 1 if alg == 'DQN' else 1

        for _ in range(avg_times):

            if alg in ['DQN', ]:
                #上层 选择编码器
                h0_agent = DQN(gamma=0.99, lr=0.0001, action_num=3, state_num=6,
                            buffer_size=10000, batch_size=64, INITIAL_EPSILON=0.2, FINAL_EPSILON=0.001,max_episode=1000,
                            replace=1000, chkpt_dir='./chkpt',)
                #下层 选择编码深度
                h1_agent = DQN(gamma=0.99, lr=0.0001, action_num=8, state_num=6,
                            buffer_size=10000, batch_size=64, INITIAL_EPSILON=0.2, FINAL_EPSILON=0.001,max_episode=1000,
                            replace=1000, chkpt_dir='./chkpt')
            else:
                fixed_mode = ['Basic', 'Residual', 'Dense'].index(alg.split('-')[0])
                max_fixed_depth = int(alg.split('-')[1])
                
                agent = FixedAgent(fixed_mode, max_fixed_depth)
            # encoder = DenseEncoder(data_depth, hidden_size).to(device)  # 编码器
            encoders = {depth: [] for depth in range(1, 1 + max_data_depth)}
            for depth in encoders.keys():
                for Encoder in [BasicEncoder, ResidualEncoder, DenseEncoder]:
                    encoders[depth].append(Encoder(int(depth), hidden_size).to(device))
            decoders = {depth: DenseDecoder(int(depth), hidden_size).to(device) for depth in range(1, 1 + max_data_depth)}
            
            critic = BasicCritic(hidden_size).to(device)  # critic评估器
            cr_optimizer = Adam(critic.parameters(), lr=1e-4)  # critic模型优化

            # s
            message_density_s = []
            cover_scores = []
            consumptions = []
            bpps = []
            encode_mse_losses = []
            uiqi_s = []
            rs_s = []
            # r
            h1_rewards = []
            h0_rewards = []
            rewards = []
            psnr_s = []
            ssim_s = []
            h0_next_state = [0, 0, 0, 0, False, 0]
            h1_next_state = [0, 0, 0, 0, False, 0]
            next_state = [0, 0, 0, 0, False, 0]

            for ep in tqdm(range(epochs)):
                # 原始 dataset 抽样到 subset
                train_size = 100
                valid_size = 50
                train_indices = np.random.choice(range(len(train_set)), train_size, replace=False)
                valid_indices = np.random.choice(range(len(valid_set)), valid_size, replace=False)
                train_subset = torch.utils.data.Subset(train_set, train_indices)
                valid_subset = torch.utils.data.Subset(valid_set, valid_indices)
                train_loader = torch.utils.data.DataLoader(train_subset, batch_size=4, shuffle=True)
                valid_loader = torch.utils.data.DataLoader(valid_subset, batch_size=4, shuffle=False)

                consumption_weight = {'encode_mode': 1, 'depth': 0.1}

                if alg in ['DQN', ]: 
                    # 更新状态 选择动作
                    h0_state = h0_next_state
                    h1_state = h1_next_state
                    h0_action, _ = h0_agent.choose_action(h0_state)
                    h1_action, _ = h1_agent.choose_action(h1_state)
                    #模型选择和嵌入深度
                    encode_mode = h0_action
                    data_depth = h1_action +1 #深度范围从1开始
                else:
                    state=next_state
                    action, _ = agent.choose_action(state)
                    encode_mode = action // max_data_depth
                    data_depth = action % max_data_depth + 1
                if alg == 'DQN' and ep / epochs > 0.3:
                    action_log['Combination'].append(h0_action*8+h1_action+1)
                    action_log['mode'].append(encode_mode)
                    action_log['depth'].append(data_depth)

                # print('\nData depth = %s, Encoder : %s' % (data_depth, ['Basic', 'Residual', 'Dense'][encode_mode]))

                encoder = encoders[data_depth][encode_mode]
                decoder = decoders[data_depth]
                en_de_optimizer = Adam(list(decoder.parameters()) +
                                    list(encoder.parameters()), lr=1e-4)  # encoder和decoder模型优化
                metrics = {field: list() for field in METRIC_FIELDS}
                # 训练Critic
                for cover, _ in train_loader:  # 从train_loader中取出图像
                    gc.collect()
                    cover = cover.to(device)  # 将cover转移到指定设备上

                    N, _, H, W = cover.size()  # 这行代码获取输入数据 cover 的形状信息，其中 N 表示批次大小，H 和 W 分别表示输入图像的高度和宽度
                    # sampled from the discrete uniform distribution over 0 to 2
                    payload = torch.zeros((N, data_depth, H, W),
                                        # 这行代码创建一个形状为 (N, data_depth, H, W) 的全零张量 payload，并随机填充为 0 或 1。
                                        device=device).random_(0, 2)
                    generated = encoder.forward(cover, payload)  # 生成隐写图片
                    cover_score = torch.mean(critic.forward(cover))  # 原始图片评估
                    generated_score = torch.mean(critic.forward(generated))  # 生成图片评估

                    cr_optimizer.zero_grad()  # Critic优化
                    (cover_score - generated_score).backward(retain_graph=False)  # 损失函数
                    cr_optimizer.step()

                    for p in critic.parameters():  # 遍历 critic 模型的参数，并将它们限制在一个范围内
                        p.data.clamp_(-0.1, 0.1)
                    metrics['train.cover_score'].append(cover_score.item())  # 添加相关参数
                    metrics['train.generated_score'].append(generated_score.item())
                # 训练编码器和解码器
                rgb_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(device)
                for cover, _ in train_loader:
                    gc.collect()
                    cover = cover.to(device)

                    def wavelet_trans(cover):
                        cover_gray = torch.sum(cover * rgb_weights, dim=1).unsqueeze(1)
                        # 利用小波变换计算图片的高频位置
                        coeffs = pywt.wavedec(cover_gray.cpu(), 'haar')
                        # 近似分量（低频）
                        approximation_coeffs = coeffs[0]
                        # 细节分量（高频）
                        detail_coeffs = coeffs[1:]
                        # 设定一个阈值，确定哪些系数可以用于隐写
                        # 这里我们选择保留一定数量的最大能量的高频系数
                        threshold = np.percentile(np.abs(coeffs[1:]).ravel(), 50)  # 例如，取前50%的能量阈值
                        significant_coeffs = np.abs(coeffs[1:]) > threshold

                        # 计算适合隐写的系数数量
                        # 这里我们统计所有三个方向的高频系数
                        stego_pixels_count = np.sum(significant_coeffs)
                    
                    N, _, H, W = cover.size()
                    # sampled from the discrete uniform distribution over 0 to 2
                    payload = torch.zeros((N, data_depth, H, W),
                                        device=device).random_(0, 2)
                    generated = encoder.forward(cover, payload)  # 生成图片
                    decoded = decoder.forward(generated)  # 隐写信息

                    UIQI = 0
                    rs = True
                    for c, g in zip(cover, generated):
                        UIQI += uiqi(c, g)
                        rs &= rs_analysis(g)
                    UIQI /= len(cover)

                    message_density = torch.sum(payload) / torch.numel(payload) # 计算消息密度

                    encoder_mse = mse_loss(generated, cover)  # 均方误差
                    decoder_loss = binary_cross_entropy_with_logits(decoded, payload)  # 交叉熵
                    decoder_acc = (decoded >= 0.0).eq(
                        payload >= 0.5).sum().float() / payload.numel()  # 解码准确率
                    generated_score = torch.mean(critic.forward(generated))  # 生成图片评估

                    en_de_optimizer.zero_grad()
                    (100.0 * encoder_mse + decoder_loss +
                    generated_score).backward()  # Why 100?   #加大均方误差的权重
                    en_de_optimizer.step()

                    metrics['train.encoder_mse'].append(encoder_mse.item())
                    metrics['train.decoder_loss'].append(decoder_loss.item())
                    metrics['train.decoder_acc'].append(decoder_acc.item())

                    metrics['train.message_density'].append(message_density.item())
                    metrics['train.UIQI'].append(UIQI.item())
                    metrics['train.rs'].append(rs)


                for cover, _ in valid_loader:
                    gc.collect()
                    cover = cover.to(device)
                    vutils.save_image(cover, "cover.png")
                    N, _, H, W = cover.size()
                    # sampled from the discrete uniform distribution over 0 to 2
                    payload = torch.zeros((N, data_depth, H, W),
                                        device=device).random_(0, 2)
                    generated = encoder.forward(cover, payload)
                    vutils.save_image(generated, "images.png")

                    decoded = decoder.forward(generated)
                    # print(decoded)
                    encoder_mse = mse_loss(generated, cover)
                    decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
                    decoder_acc = (decoded >= 0.0).eq(
                        payload >= 0.5).sum().float() / payload.numel()
                    generated_score = torch.mean(critic.forward(generated))
                    cover_score = torch.mean(critic.forward(cover))

                    metrics['val.encoder_mse'].append(encoder_mse.item())
                    metrics['val.decoder_loss'].append(decoder_loss.item())
                    metrics['val.decoder_acc'].append(decoder_acc.item())
                    metrics['val.cover_score'].append(cover_score.item())
                    metrics['val.generated_score'].append(generated_score.item())
                    metrics['val.ssim'].append(
                        pytorch_ssim.ssim(cover, generated).item())
                    metrics['val.psnr'].append(
                        10 * torch.log10(4 / encoder_mse).item())
                    metrics['val.bpp'].append(
                        data_depth * (2 * decoder_acc.item() - 1))

                encode_mse_losses.append(np.mean(metrics['val.encoder_mse']))
                bpps.append(np.mean(metrics['val.bpp']))
                cover_scores.append(np.mean(metrics['val.cover_score']))
                message_density_s.append(np.mean(metrics['train.message_density']))
                uiqi_s.append(np.mean(metrics['train.UIQI']))
                consumption = encode_mode * consumption_weight['encode_mode'] + data_depth * consumption_weight['depth']
                consumptions.append(consumption)
                psnr_s.append(np.mean(metrics['val.psnr']))
                ssim_s.append(np.mean(metrics['val.ssim']))
                rs_s.append(np.mean(metrics['train.rs']))
                #上层和下层奖励应该不同吧 
                reward = compute_reward(psnr_s[-1], ssim_s[-1], consumption)
                h0_reward=reward
                h1_reward=reward
                
                if alg in ['DQN', ]: 
                    # 计算奖励
                    h0_rewards.append(h0_reward)
                    h1_rewards.append(h1_reward)
                    # 更新状态
                    h0_next_state = formulate_state(bpps[-1], encode_mse_losses[-1], cover_scores[-1], message_density_s[-1], uiqi_s[-1], rs_s[-1])
                    h1_next_state = formulate_state(bpps[-1], encode_mse_losses[-1], cover_scores[-1], message_density_s[-1], uiqi_s[-1], rs_s[-1])
                    # 存储经验
                    h0_agent.store_transition(h0_state, h0_action, h0_reward, h0_next_state)
                    h1_agent.store_transition(h1_state, h1_action, h1_reward, h1_next_state)
                    # 学习
                    h0_agent.learn()
                    h1_agent.learn()
                else:
                    rewards.append(reward)
                    next_state = formulate_state(bpps[-1], encode_mse_losses[-1], cover_scores[-1], message_density_s[-1], uiqi_s[-1], rs_s[-1])
                    agent.store_transition(state, action, reward, next_state)
                    agent.learn()

                # now = datetime.datetime.now()
                # 保存模型状态
                name = "EN_DE_%+.3f.dat" % (cover_score.item())
                fname = os.path.join('.', 'results','model', name)
                states = {
                    'state_dict_critic': critic.state_dict(),
                    'state_dict_encoder': encoder.state_dict(),
                    'state_dict_decoder': decoder.state_dict(),
                    'en_de_optimizer': en_de_optimizer.state_dict(),
                    'cr_optimizer': cr_optimizer.state_dict(),
                    'metrics': metrics,
                    'train_epoch': ep,
                    # 'date': now.strftime("%Y-%m-%d_%H:%M:%S"),
                }
                torch.save(states, fname)
                # 指标
                # plot('encoder_mse', ep, metrics['val.encoder_mse'], True)
                # plot('decoder_loss', ep, metrics['val.decoder_loss'], True)
                # plot('decoder_acc', ep, metrics['val.decoder_acc'], True)
                # plot('cover_score', ep, metrics['val.cover_score'], True)
                # plot('generated_score', ep, metrics['val.generated_score'], True)
                # plot('ssim', ep, metrics['val.ssim'], True)
                # plot('psnr', ep, metrics['val.psnr'], True)
                # plot('bpp', ep, metrics['val.bpp'], True)

        # for data, y_label in zip([encode_mse_losses, bpps, rewards, psnr_s, ssim_s, consumptions],
        #                         ['Encoder mse loss ', 'Bits per pixel', 'Reward', 'PSNR', 'SSIM', 'Consumption', ]):
            h0_reward_avg.append(h0_rewards)
            h1_reward_avg.append(h1_rewards)
            reward_avg.append(rewards)
            mse_avg.append(encode_mse_losses)
            psnr_avg.append(psnr_s)
            ssim_avg.append(ssim_s)
            consumption_avg.append(consumptions)
            uiqi_avg.append(uiqi_s)
            rs_avg.append(rs_s)

        h0_reward_avg = np.mean(h0_reward_avg, axis=0).tolist()
        h1_reward_avg = np.mean(h1_reward_avg, axis=0).tolist()
        reward_avg = np.mean(reward_avg, axis=0).tolist()
        psnr_avg = np.mean(psnr_avg, axis=0).tolist()
        ssim_avg = np.mean(ssim_avg, axis=0).tolist()
        uiqi_avg = np.mean(uiqi_avg, axis=0).tolist()
        consumption_avg = np.mean(consumption_avg, axis=0).tolist()
        rs_avg = np.mean(rs_avg, axis=0).tolist()


        

        # 分析 DQN 收敛时选择的动作
        if alg == 'DQN':
            from collections import Counter
            for key, title in zip(action_log.keys(), ['Encode mode', 'Embed depth', 'Combination']):
                element_counts = Counter(action_log[key])
                plt.figure()
                plt.pie(element_counts.values(), labels=element_counts.keys(), autopct='%1.1f%%')
                # plt.axis('equal')  # 确保饼图是圆形的
                plt.title(title)
                # plt.tight_layout()
                plt.savefig(os.path.join('.', 'results', data_dir, 'Action_' + title + '.png'))
                plt.close()

            for data, y_label in zip([h0_reward_avg, h1_reward_avg,psnr_avg, ssim_avg, consumption_avg, uiqi_avg, rs_avg, mse_avg],
                                ['H0_Reward','H1_Reward', 'PSNR', 'SSIM', 'Consumption', 'UIQI', 'RS test','MSE loss']):
                plt.figure()
                plt.plot(data)
                plt.xlabel('Time slot')
                plt.ylabel(y_label)
                plt.title(alg)
                plt.tight_layout()
                plt.savefig(os.path.join('.', 'results', data_dir , alg + '_' + y_label + '.png'))
                plt.close()

        h0_reward_alg[alg] = h0_reward_avg
        h1_reward_alg[alg] = h1_reward_avg
        reward_alg[alg] = reward_avg
        psnr_alg[alg] = psnr_avg
        ssim_alg[alg] = ssim_avg
        consumption_alg[alg] = consumption_avg
        uiqi_alg[alg] = uiqi_avg
        rs_alg[alg] = rs_avg
        mse_alg[alg] = mse_avg
    if alg == 'DQN':
        for data, y_label in zip([h0_reward_alg,h0_reward_alg,h1_reward_alg ,psnr_alg, ssim_alg, consumption_alg, uiqi_alg, rs_alg, mse_alg], ['H0_Reward','H0_Reward','H1_Reward', 'PSNR', 'SSIM', 'Consumption', 'UIQI', 'RS test', 'MSE loss']):
            # 上面多设置一次h0_reward_alg是为了能和其它算法一起画图
            plt.figure()
            for alg in algs:
                plt.plot(data[alg], label=alg.split('-')[0] if alg != 'DQN' else alg)
            plt.xlabel('Time slot')
            plt.ylabel(y_label)
            plt.legend()
            plt.title(data_dir.upper())
            plt.tight_layout()
            # if not os.path.exists(os.path.join('.', 'results/', data_dir)):
            #     os.mkdir(os.path.join('.', 'results/', data_dir))
            plt.savefig(os.path.join('.', 'results', data_dir,  'All_' + y_label + '.png'))
            plt.close()
    else:
        for data, y_label in zip([reward_alg, psnr_alg, ssim_alg, consumption_alg, uiqi_alg, rs_alg, mse_alg], ['Reward', 'PSNR', 'SSIM', 'Consumption', 'UIQI', 'RS test', 'MSE loss']):
            plt.figure()
            for alg in algs:
                plt.plot(data[alg], label=alg.split('-')[0] if alg != 'DQN' else alg)
            plt.xlabel('Time slot')
            plt.ylabel(y_label)
            plt.legend()
            plt.title(data_dir.upper())
            plt.tight_layout()
            # if not os.path.exists(os.path.join('.', 'results/', data_dir)):
            #     os.mkdir(os.path.join('.', 'results/', data_dir))
            plt.savefig(os.path.join('.', 'results', data_dir,  'All_' + y_label + '.png'))
            plt.close()

    # save mat
    import scipy.io as sio
    sio.savemat('results.mat', {'reward': reward_alg, 'psnr': psnr_alg, 'ssim': ssim_alg, 'consumption': consumption_alg, 'uiqi': uiqi_alg, 'rs': rs_alg})


if __name__ == '__main__':
    for func in [
        lambda: os.mkdir(os.path.join('.', 'results')) if not os.path.exists(os.path.join('.', 'results')) else None,
        lambda: os.mkdir(os.path.join('.', 'results','model')) if not os.path.exists(
            os.path.join('.', 'results','model')) else None,
        lambda: os.mkdir(os.path.join('.', 'results','plots')) if not os.path.exists(
            os.path.join('.', 'results','plots')) else None]:  # create directories
        try:
            func()
        except Exception as error:
            print(error)
            continue
    main()
