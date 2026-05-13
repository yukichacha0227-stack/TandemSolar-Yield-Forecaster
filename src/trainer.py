def train_with_logging(model, train_loader, val_loader, epochs=100):
    """「進捗率」「Loss 8桁」を表示する標準トレーニング関数"""
    best_loss = float('inf')
    for epoch in range(epochs):
        # ...学習処理...
        train_loss = 0.00001234 # ダミー
        val_loss = 0.00001111   # ダミー
        
        progress = (epoch + 1) / epochs * 100
        log_msg = f"[{progress:3.0f}%] Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}"
        
        if val_loss < best_loss:
            best_loss = val_loss
            log_msg += " ★ Best Val Loss Updated!"
            # 保存処理など
        print(log_msg)
