import torch, numpy as np, sys
sys.path.insert(0, '.')

ckpt = torch.load('models/best_run5_forensic.pth', map_location='cpu', weights_only=False)
print('=== best_run5_forensic.pth ===')
print('  Epoch:', ckpt.get('epoch', 'N/A'), '(0-indexed, yani epoch', ckpt.get('epoch', 0) + 1, ')')
print('  AUC:', round(ckpt.get('val_auc', 0), 4))
print('  Accuracy:', round(ckpt.get('val_acc', 0), 4))
print('  F1:', round(ckpt.get('val_macro_f1', 0), 4))

ckpt5 = torch.load('models/checkpoint_epoch5.pth', map_location='cpu', weights_only=False)
print('\n=== checkpoint_epoch5.pth ===')
print('  Epoch:', ckpt5.get('epoch', 'N/A'))
print('  Best AUC:', round(ckpt5.get('best_auc', 0), 4))

cm = np.load('logs/run4/confusion_matrix_latest.npy')
print('\n=== Son Confusion Matrix ===')
print(cm)
tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
print(f'  REAL recall: {tn/(tn+fp)*100:.1f}%')
print(f'  FAKE recall: {tp/(tp+fn)*100:.1f}%')
print(f'  Toplam: {cm.sum()}')
