import sys
import numpy 
import argparse
import associate
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    # 设置字体类型
    "axes.unicode_minus": False, #解决负号无法显示的问题
    "font.size": 6
}

rcParams.update(config)


def align(model,data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    """

    numpy.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)
    
    W = numpy.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity( 3 ))
    if(numpy.linalg.det(U) * numpy.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    rotmodel = rot*model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += numpy.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
        normi = numpy.linalg.norm(model_zerocentered[:,column])
        norms += normi*normi

    s = float(dots/norms)    
    
    transGT = data.mean(1) - s*rot * model.mean(1)
    trans = data.mean(1) - rot * model.mean(1)

    model_alignedGT = s*rot * model + transGT
    model_aligned = rot * model + trans

    alignment_errorGT = model_alignedGT - data
    alignment_error = model_aligned - data



    trans_errorGT = numpy.sqrt(numpy.sum(numpy.multiply(alignment_errorGT,alignment_errorGT),0)).A[0]
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error,alignment_error),0)).A[0]
        
    return rot,transGT,trans_errorGT,trans,trans_error, s, alignment_errorGT

def plot_traj(ax,stamps,traj,style,color,label):
    """
    Plot a trajectory using matplotlib. 
    
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    stamps.sort()
    interval = numpy.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x)>0:
            ax.plot(x,y,style,color=color,label=label)
            label=""
            x=[]
            y=[]
        last= stamps[i]
    if len(x)>0:
        ax.plot(x,y,style,color=color,label=label)

def evaluate_trajectories(first_file, second_file, offset, scale, max_difference, save_path=None):
    firsttemp_list = associate.read_file_list(first_file, False)
    first_list = {float(f"{key/1e9:.6f}"): value for key, value in firsttemp_list.items()}
    second_list = associate.read_file_list(second_file, False)

    matches = associate.associate(first_list, second_list, float(offset), float(max_difference))    
    if len(matches) < 2:
        print(f"Couldn't find matching timestamp pairs between {first_file} and {second_file}!")
        return

    first_xyz = numpy.matrix([[float(value) for value in first_list[a][0:3]] for a,b in matches]).transpose()
    second_xyz = numpy.matrix([[float(value)*float(scale) for value in second_list[b][0:3]] for a,b in matches]).transpose()
    dictionary_items = second_list.items()
    sorted_second_list = sorted(dictionary_items)

    second_xyz_full = numpy.matrix([[float(value)*float(scale) for value in sorted_second_list[i][1][0:3]] for i in range(len(sorted_second_list))]).transpose()

    rot, transGT, trans_errorGT, trans, trans_error, scale, alignment_errorGT = align(second_xyz, first_xyz)
    if save_path:
        with open(save_path, 'w') as f:
            for error in trans_error:
                f.write(str(error) + '\n')

    # x = range(len(trans_error))
    # plt.plot(x, trans_error, linestyle='-')
    # plt.title('Translational Error')
    # plt.xlabel('Index')
    # plt.ylabel('Error')
    # plt.grid(True)
    # plt.show()

    print("Evaluation completed for", second_file)
    print("compared_pose_pairs %d pairs" % len(trans_error))
    print("absolute_translational_error.rmse %f m" % numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error)))
    print("absolute_translational_error.mean %f m" % numpy.mean(trans_error))
    print("absolute_translational_error.median %f m" % numpy.median(trans_error))
    print("absolute_translational_error.std %f m" % numpy.std(trans_error))
    print("absolute_translational_error.min %f m" % numpy.min(trans_error))
    print("absolute_translational_error.max %f m" % numpy.max(trans_error))
    print("max idx: %i" % numpy.argmax(trans_error))
    return alignment_errorGT

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory.') 
    parser.add_argument('first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('second_files', nargs='+', help='list of estimated trajectories (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)', default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 10000000 ns)', default=0.2)
    parser.add_argument('--save_path', help='directory to save the error results', default=None)
    args = parser.parse_args()

    fig, ax = plt.subplots(3, 1, figsize=(5, 2.3))
    #color = [()]
    aixname = ['x', 'y','z']
    i = 0
    for second_file in args.second_files:
        if i == 0:
            col= '#63B2EE'
            col= (230/255,111/255,91/255)
            col= '#2878B5'
            
            col='#e7724f'
            col='#0000FF'
            i = i+1
        else:
            col = '#f8cb7f'
            col= (42/255,157/255,142/255)
            col= '#FA7F6F'
            col='#0000FF'
            col = '#ff8ba7'
        save_file = None
        if args.save_path:
            save_file = f"{args.save_path}/{second_file.split('/')[-1].replace('.txt', '')}_error.txt"
        alignment_errorGT = evaluate_trajectories(args.first_file, second_file, args.offset, args.scale, args.max_difference, save_file)
        num_dimensions, num_points = alignment_errorGT.shape
        x = numpy.arange(num_points)
    
        for dim in range(num_dimensions):
        # 获取当前维度的数据
            y = numpy.array(alignment_errorGT[dim, :]).reshape(-1)
            y = numpy.abs(y)
        # 绘制折线图
            #ax[dim].plot(x, y, linestyle='-', color=(114/255, 188/255, 213/255))
            ax[dim].plot(x, y, color=col, linewidth=0.9)
        # 添加标题和标签
            ax[0].set_title('Error in V202 Sequence')
            # ax[dim].set_xlabel('Index')
            ax[dim].set_ylabel(f'{aixname[dim]} Error[m]')
        # 添加图例-legend
            ax[0].legend(['Rover-SLAM','ORB-SLAM3'],loc='upper right', bbox_to_anchor=(1.0, 1.0))
    # 显示网格
            # ax[dim].grid(True)
    plt.tight_layout()

    plt.savefig('plot_2.pdf', format='pdf')
    plt.show()
