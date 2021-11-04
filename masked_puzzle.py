#%%
import itertools
import numpy as np
import torch
import random
import sys
import os
import torchvision
import time

#%%
setting = {
    'base' : 14,
    'panels' : None,
    'loader' : None,
}

#%%
def normalize(image):
    # into 0-1 range
    if image.max() == image.min():
        return image - image.min()
    else:
        return (image - image.min())/(image.max() - image.min())

def equalize(image):
    from skimage import exposure
    return exposure.equalize_hist(image)

def enhance(image):
    return np.clip((image-0.5)*3,-0.5,0.5)+0.5

def preprocess(image):
    image = image.astype(float)
    image = equalize(image)
    image = normalize(image)
    image = enhance(image)
    return image

def split_image(path,width,height):
    from scipy import misc
    img = misc.imread(path,True)/256
    # convert the image to *greyscale*
    W, H = img.shape
    dW, dH = W//width, H//height
    img = img[:height*dH,:width*dW]
    return np.transpose(img.reshape((height,dH,width,dW)), (0,2,1,3)).reshape((height*width,dH,dW))

#%%
def mnist_panels():
    x_train, y_train = torch.load('data/mnist-data/processed/training.pt')
    filters = [ np.equal(i, y_train) for i in range(9)]
    imgs = [x_train[f] for f in filters]
    panels = [i_imgs[0].reshape((28, 28)).numpy() for i_imgs in imgs]
    panels[8] = imgs[8][3].reshape((28, 28)).numpy()
    panels[1] = imgs[1][3].reshape((28, 28)).numpy()

    panels = np.array(panels)
    panels = (panels.astype('float32') / 255.).round()
    base = setting['base']
    stepy = panels.shape[1] // base
    stepx = panels.shape[2] // base

    panels = panels[:, ::stepy, ::stepx][:, :base, :base].round()
    panels = preprocess(panels)
    
    return panels

def mandrill_panels(width, height):
    base = setting['base']
    # import ipdb
    # ipdb.set_trace()
    panels = split_image("data/mandrill.bmp",width,height)
    stepy = panels.shape[1]//base
    stepx = panels.shape[2]//base
    panels = panels[:,:stepy*base,:stepx*base,]
    panels = panels.reshape((panels.shape[0],base,stepy,base,stepx))
    panels = panels.mean(axis=(2,4))
    panels = preprocess(panels)
    return panels

def spider_panels(width, height):
    base = setting['base']
    panels = split_image("data/spider.png",width,height)
    stepy = panels.shape[1]//base
    stepx = panels.shape[2]//base
    panels = panels[:,:stepy*base,:stepx*base,]
    panels = panels.reshape((panels.shape[0],base,stepy,base,stepx))
    panels = panels.mean(axis=(2,4))
    panels = preprocess(panels)
    return panels

#%%
def load(domain, width, height):
    if setting['panels'] is None:
        if domain == 'mandrill':
            # import ipdb
            # ipdb.set_trace()
            setting['panels'] = mandrill_panels(width, height)
        elif domain == 'spider':
            setting['panels'] = spider_panels(width, height)
        else:
            print('domain not found, use \"mnist\".')
            setting['panels'] = mnist_panels()

    return setting['panels']

#%%
def puzzle_generate(states, width, height, domain):
    # latplan version - state[num] = pos
    base = setting['base']
    panels = load(domain, width, height).astype('float32')
    P = len(panels)
    
    states_one_hot = torch.zeros(*states.shape, width*height).scatter_(2, torch.tensor(states).reshape([*states.shape, 1]), 1)
    matches = states_one_hot.transpose(1,2).reshape([-1, P])
    panels = torch.from_numpy(panels).reshape([P, base*base])
    observations = torch.matmul(matches, panels).reshape([-1, height, width, base, base]).permute([0,1,3,2,4]).reshape([-1, height*base, width*base])

    return observations

def puzzle_generate_v2(states, width, height, domain):
    # poplan version - state[pos] = num
    # print('1')
    # import ipdb
    # ipdb.set_trace()
    base = setting['base']
    panels = load(domain, width, height).astype('float32')
    P = len(panels)
    
    states_one_hot = torch.zeros(*states.shape, width*height).scatter_(-1, torch.tensor(states).reshape([*states.shape, 1]), 1)
    matches = states_one_hot
    panels = torch.from_numpy(panels).reshape([P, base*base])
    observations = torch.matmul(matches, panels).reshape([-1, height, width, base, base]).permute([0,1,3,2,4]).reshape([*states.shape[:-1], height*base, width*base])

    return observations

#%%
### Generate mask
def manmade_blocks(mask_pos=[0,4], width=3, height=3):
    base = setting['base']
    mask_mat = np.ones([height*base, width*base], dtype='float32')
    for pos in mask_pos:
        assert pos >= 0
        assert pos < width*height
        row, col = pos // width, pos % width
        row_s = row*base
        row_e = (row+1)*base
        col_s = col*base
        col_e = (col+1)*base
        # print(row_s, row_e, col_s, col_e)
        mask_mat[row_s: row_e, col_s:col_e] = 0
    return mask_mat

def mask_random(num_blocks,blocks_size):
    start_time = time.time()
    mask_mat = np.ones([42,42], dtype='float32')
    for i in range(num_blocks):
        # print('--'*10,i)

        # if not os.path.exists('pic_explore'):
        #     os.makedirs('pic_explore')
        # torchvision.utils.save_image(torch.from_numpy( mask_mat), 'pic_explore/'+str(i)+'.png')
        t = 0
        x_axis = random.randint(0,41-blocks_size+1)
        if random.randint(0,1) == 1:
            y_axis = np.argmax(mask_mat[min(41-blocks_size+1,x_axis)]) + random.randint(0,blocks_size)
        else:
            y_axis = random.randint(0,41-blocks_size+1)
        # x_axis = 0
        while np.sum(mask_mat[x_axis:x_axis+blocks_size,y_axis:y_axis+blocks_size] ) != blocks_size**2:
            # print(t)
            # print(x_axis,y_axis)
            # import ipdb
            # ipdb.set_trace()
            direction = random.randint(0,3) ## 0 下 1 上 2 左 3 右
            # interval = random.randint(0,(41 - max(x_axis,y_axis)))
            interval = blocks_size + random.randint(0,3)
            # interval = blocks_size + 1
            if direction == 0:
                y_axis = min(41-blocks_size+1,y_axis + interval)
            if direction == 1:
                y_axis = max(0,y_axis - interval)
            if direction == 2:
                x_axis = max(0,x_axis - interval)
            if direction == 3:
                x_axis = min(41-blocks_size+1,x_axis + interval)
            t+=1
            if t == 6:
                check = random.randint(0,2)
                t = 0
                if check  == 0:
                    for l in range(0,41-blocks_size+1):
                        if np.sum(mask_mat[x_axis:x_axis+blocks_size,l:l+blocks_size] ) != blocks_size**2:
                            y_axis = l
                    # y_axis = np.argmax(mask_mat[min(41-blocks_size+1,x_axis)])
                if check == 1:
                    for l in range(0,41-blocks_size+1):
                        if np.sum(mask_mat[l:l+blocks_size,y_axis:y_axis+blocks_size] ) != blocks_size**2:
                            y_axis = l
                    # x_axis = np.argmax(mask_mat[:,min(41-blocks_size+1,y_axis)])
                if check == 2:
                    if random.randint(0,1)  == 0:
                        x_axis = random.randint(0,41-blocks_size+1)
                    if random.randint(0,1)  == 1:
                        y_axis = random.randint(0,41-blocks_size+1)
                # if check == 3: 
                #     x_axis = random.randint(0,41-blocks_size+1)
                #     y_axis = random.randint(0,41-blocks_size+1)
            if time.time() - start_time > 10:
                return mask_mat,False            
        
    return mask_mat,True

def manmade_blocks_new(num_blocks,blocks_size):
    x_length = 42
    y_length = 42
    if blocks_size == 1000:
        mask_mat = np.ones([x_length,y_length], dtype='float32')
        for i in range(num_blocks):
            x_axis = random.randint(0,x_length-1-blocks_size+1)
            y_axis = random.randint(0,y_length-1-blocks_size+1)
            t = 0
            while np.sum(mask_mat[x_axis:x_axis+blocks_size,y_axis:y_axis+blocks_size] ) != blocks_size**2 and t<5:
                x_axis = random.randint(0,x_length-1-blocks_size+1)
                y_axis = random.randint(0,y_length-1-blocks_size+1)
                t += 1
            mask_mat[x_axis:x_axis+blocks_size,y_axis:y_axis+blocks_size] = 0
        return mask_mat
    else:
        start_time = time.time()
        mask_mat = np.ones([x_length,y_length], dtype='float32')
        for i in range(num_blocks):
            # print('--'*10,i)

            # if not os.path.exists('pic_explore'):
            #     os.makedirs('pic_explore')
            # torchvision.utils.save_image(torch.from_numpy( mask_mat), 'pic_explore/'+str(i)+'.png')
            t = 0
            x_axis = random.randint(0,x_length-1-blocks_size+1)
            if random.randint(0,1) == 1:
                y_axis = np.argmax(mask_mat[min(y_length-1-blocks_size+1,x_axis)]) + random.randint(0,blocks_size)
            else:
                y_axis = random.randint(0,y_length-1-blocks_size+1)
            # x_axis = 0
            while np.sum(mask_mat[x_axis:x_axis+blocks_size,y_axis:y_axis+blocks_size] ) != blocks_size**2:
                # print(t)
                # print(x_axis,y_axis)
                # import ipdb
                # ipdb.set_trace()
                direction = random.randint(0,3) ## 0 下 1 上 2 左 3 右
                # interval = random.randint(0,(41 - max(x_axis,y_axis)))
                interval = blocks_size + random.randint(0,3)
                # interval = blocks_size + 1
                if direction == 0:
                    y_axis = min(y_length-1-blocks_size+1,y_axis + interval)
                if direction == 1:
                    y_axis = max(0,y_axis - interval)
                if direction == 2:
                    x_axis = max(0,x_axis - interval)
                if direction == 3:
                    x_axis = min(x_length-1-blocks_size+1,x_axis + interval)
                t+=1
                if t == 6:
                    check = random.randint(0,2)
                    t = 0
                    if check  == 0:
                        for l in range(0,y_length-1-blocks_size+1):
                            if np.sum(mask_mat[x_axis:x_axis+blocks_size,l:l+blocks_size] ) != blocks_size**2:
                                y_axis = l
                        # y_axis = np.argmax(mask_mat[min(41-blocks_size+1,x_axis)])
                    if check == 1:
                        for l in range(0,y_length-1-blocks_size+1):
                            if np.sum(mask_mat[l:l+blocks_size,y_axis:y_axis+blocks_size] ) != blocks_size**2:
                                y_axis = l
                        # x_axis = np.argmax(mask_mat[:,min(41-blocks_size+1,y_axis)])
                    if check == 2:
                        if random.randint(0,1)  == 0:
                            x_axis = random.randint(0,x_length-blocks_size+1)
                        if random.randint(0,1)  == 1:
                            y_axis = random.randint(0,y_length-1-blocks_size+1)
                    # if check == 3: 
                    #     x_axis = random.randint(0,41-blocks_size+1)
                    #     y_axis = random.randint(0,41-blocks_size+1)
                    # if check == 4:
                    #     x_axis = 0
                    #     y_axis = 0

                if time.time() - start_time > 10:
                    new_mask,check = mask_random(num_blocks,blocks_size)
                    while check == False:
                        new_mask,check = mask_random(num_blocks,blocks_size)
                    return new_mask
                
            mask_mat[x_axis:x_axis+blocks_size,y_axis:y_axis+blocks_size] = 0
        return mask_mat
def mask_generation(size=14, num=2):
    # TODO: 参考MisGAN
    NotImplemented

#%%
def mask(observations, mask_type='manmade_blocks', tau=0.5, **kwargs):
    if mask_type == 'manmade_blocks':
        mask_mat = torch.tensor(manmade_blocks(**kwargs))
        # print(torch.max(mask_mat), torch.min(mask_mat))
    
    return observations*mask_mat + tau*(1-mask_mat)


#%%
def generate_states(num_digit=9):
    return itertools.permutations(range(num_digit))

def successors_with_act(state, width, height):
    pos = state[0]
    x = pos % width
    y = pos // width
    succ = []
    try:
        if x != 0:
            act = 1     # left
            s = list(state)
            other = next(i for i,_pos in enumerate(s) if _pos == pos-1)
            s[0] -= 1
            s[other] += 1
            succ.append([s, act])
        if x != width-1:
            act = 2     # right
            s = list(state)
            other = next(i for i,_pos in enumerate(s) if _pos == pos+1)
            s[0] += 1
            s[other] -= 1
            succ.append([s, act])
        if y != 0:
            act = 3     # up
            s = list(state)
            other = next(i for i,_pos in enumerate(s) if _pos == pos-width)
            s[0] -= width
            s[other] += width
            succ.append([s, act])
        if y != height-1:
            act = 4     # down
            s = list(state)
            other = next(i for i,_pos in enumerate(s) if _pos == pos+width)
            s[0] += width
            s[other] -= width
            succ.append([s, act])
        return succ
    except StopIteration:
        board = np.zeros((height,width))
        for i in range(height*width):
            _pos = state[i]
            _x = _pos % width
            _y = _pos // width
            board[_y,_x] = i
        print(board)
        print(succ)
        print(act)
        print((s,x,y,width,height))

#%%
def traces(width, height, states=None, trace_num=32, trace_len=14, action_dim=4, **kwargs):
    num_digit = width * height
    if states is None:
        states = generate_states(num_digit)
        states = np.array([s for s in states])
        np.random.shuffle(states)
    pre_states = np.array([state for state in itertools.islice(states, trace_num)])
    
    assert len(pre_states) == trace_num
    observations = puzzle_generate(pre_states, width, height)
    observations = np.expand_dims(observations, 1)
    actions = np.zeros([len(pre_states), 1, action_dim])

    def pickone(arr):
        return arr[np.random.randint(0, len(arr))]
    def one_hot(arr, dim=4):
        return (np.arange(dim) == arr[:, None]-1).astype(np.float32)

    for i in range(trace_len-1):

        suc_list = [pickone(successors_with_act(state, width, height)) for state in pre_states]
        suc_states = np.array(list(map(lambda x: x[0], suc_list)))
        curr_act = one_hot(np.array(list(map(lambda x: x[1], suc_list))), action_dim)
        suc_obs = puzzle_generate(suc_states, width, height, **kwargs)
        observations = np.concatenate((observations, np.expand_dims(suc_obs, 1)), axis=1)
        actions = np.concatenate((actions, np.expand_dims(curr_act, 1)), axis=1)
        pre_states = suc_states 
    
    return observations, actions


# %%
def test_masked_puzzle():       
    states = np.array([[0,1,2,3,4,5,6,7,8], [1,2,3,4,8,7,6,5,0], [8,7,6,5,1,2,3,4,0], [8,7,6,5,4,3,2,1,0], [3,2,1,0,4,5,6,8,7]])
    observations = puzzle_generate(states, 3, 3)
    observations = torch.stack([observations, torch.transpose(observations, 1, 2)])
    masked_obs = mask(observations)

    import matplotlib.pyplot as plt
    # import pylab as plt
    import pylab as plt
    fig = plt.figure()
    l = len(states)
    for i in range(l):
        print(states[i])
        ax1 = fig.add_subplot(4, l, i+1)
        ax1.imshow(observations[0][i], cmap=plt.cm.get_cmap("gray"))
        ax2 = fig.add_subplot(4, l, l+i+1)
        ax2.imshow(masked_obs[0][i], cmap=plt.cm.get_cmap("gray"))
        ax3 = fig.add_subplot(4, l, 2*l+i+1)
        ax3.imshow(observations[1][i], cmap=plt.cm.get_cmap("gray"))
        ax4 = fig.add_subplot(4, l, 3*l+i+1)
        ax4.imshow(masked_obs[1][i], cmap=plt.cm.get_cmap("gray"))
        plt.tight_layout()

#%%
def test_traces():
    width=3
    height=3
    states=None
    trace_num=2
    trace_len=3
    action_dim=4
    obs, actions = traces(width, height, states, trace_num, trace_len, action_dim)
    masked_obs = mask(obs)


    # import matplotlib.pyplot as plt
    import pylab as plt
    fig = plt.figure()
    for i in range(trace_num):
        print(actions[i])
        for j in range(trace_len):
            ax1 = fig.add_subplot(2*trace_num, trace_len, 2*i*trace_len+j+1)
            ax1.imshow(obs[i][j], cmap=plt.cm.get_cmap("gray"))
            ax2 = fig.add_subplot(2*trace_num, trace_len, (2*i+1)*trace_len+j+1)
            ax2.imshow(masked_obs[i][j], cmap=plt.cm.get_cmap("gray"))

# %%
if __name__ == "__main__":
    test_traces()

# %%
