import coviar_py2
import coviar_all

video_dir = 'Two_Towers_6_jump_f_nm_np1_le_bad_8.mp4'
gop_id = 1
pos_id = 9
im_py2 = coviar_py2.load(video_dir, gop_id, pos_id, 0, True)
mv_py2 = coviar_py2.load(video_dir, gop_id, pos_id, 1, True)
res_py2 = coviar_py2.load(video_dir, gop_id, pos_id, 2, True)
res_mv_all = coviar_all.load(video_dir, gop_id, pos_id, 2, True)
im_all = res_mv_all[0]
res_all = res_mv_all[1]
mv_all = res_mv_all[2]
print (im_all == im_py2).all()
print (res_all == res_py2).all()
print (mv_all == mv_py2).all()
import pdb;pdb.set_trace()
