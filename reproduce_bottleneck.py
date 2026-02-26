
import time
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
from utility import DSSDatasets
from models.DSS import DSS
import cProfile
import pstats
import io
import os

def main():
    # 1. 설정 파일 로드 (Load Config)
    conf = yaml.safe_load(open("./config.yaml"))
    
    # NetEase_DSS 설정 사용
    dataset_name = "NetEase_DSS"
    if dataset_name not in conf:
        print(f"Error: {dataset_name} not found in config.yaml")
        return

    conf = conf[dataset_name]
    conf["dataset"] = dataset_name
    conf["gpu"] = "0"
    
    # 2. 데이터셋 로드 (Load Dataset)
    print("데이터셋 로딩 중... (Loading dataset...)")
    start_time = time.time()
    dataset = DSSDatasets(conf)
    print(f"데이터셋 로딩 완료: {time.time() - start_time:.2f}초")
    
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items
    
    # 3. 하이퍼파라미터 리스트에서 단일 값 추출 (Fix Config Lists to Scalars)
    # 모델은 리스트가 아니라 단일 int/float 값을 기대하므로 변환이 필요합니다.
    conf["l2_reg"]       = conf["l2_regs"][0]
    conf["embedding_size"] = conf["embedding_sizes"][0]
    conf["UB_ratio"]     = conf["UB_ratios"][0]
    conf["UI_ratio"]     = conf["UI_ratios"][0]
    conf["BI_ratio"]     = conf["BI_ratios"][0]
    conf["num_layers"]   = conf["num_layers_options"][0]  # num_layerss -> num_layers_options로 변경됨
    conf["c_lambda"]     = conf["c_lambdas"][0]
    conf["c_temp"]       = conf["c_temps"][0]
    
    # 디바이스 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    
    # 4. 모델 및 옵티마이저 초기화 (Init Model)
    conf["aug_type"] = "None"  # <--- 노이즈 제거 (Noise Disabled)
    print("노이즈 증강(Noise Augmentation)을 끄고 테스트합니다.")

    model = DSS(conf, dataset.graphs, dataset.bundle_info).to(device)
    optimizer = optim.Adam(model.parameters(), lr=conf["lrs"][0], weight_decay=conf["l2_reg"])
    
    print("프로파일링 시작... (Starting profiling loop...)")
    
    # 5. 프로파일링 (Profiling)
    pr = cProfile.Profile()
    pr.enable()
    
    model.train(True)
    batch_cnt = len(dataset.train_loader)
    print(f"Epoch 당 배치 수: {batch_cnt}")
    
    # 시간 절약을 위해 50개 배치만 실행
    limit_batches = 50
    
    start_epoch = time.time()
    for batch_i, batch in enumerate(dataset.train_loader):
        if batch_i >= limit_batches:
            break
            
        dataset_time = time.time()
        
        optimizer.zero_grad()
        batch = [x.to(device) for x in batch]
        
        # Forward Pass 시간 측정
        t0 = time.time()
        bpr_loss, c_loss = model(batch, ED_drop=False)
        forward_time = time.time() - t0
        
        loss = bpr_loss # 프로파일링을 위해 단순화
        
        # Backward Pass 시간 측정
        t0 = time.time()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - t0
        
        # GPU 동기화 대기 시간 측정 (Synchronization Overhead)
        t0 = time.time()
        l_item = loss.item() # .item() 호출 시 GPU 연산 종료를 기다림
        sync_time = time.time() - t0
        
        if batch_i % 10 == 0:
            print(f"Batch {batch_i}: Fwd={forward_time:.4f}s, Bwd={backward_time:.4f}s, Sync={sync_time:.4f}s")
            
    pr.disable()
    print(f"50 배치 실행 소요 시간: {time.time() - start_epoch:.2f}초")
    
    # 6. 결과 출력 (Print Stats)
    s = io.StringIO()
    sortby = 'cumulative' # 누적 시간 순으로 정렬
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20) # 상위 20개만 출력
    print(s.getvalue())

if __name__ == "__main__":
    main()
