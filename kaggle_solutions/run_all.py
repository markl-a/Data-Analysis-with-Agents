"""
運行所有Kaggle解決方案的腳本
"""
import os
import sys
import subprocess
from pathlib import Path


def find_all_solutions():
    """找出所有solution.py文件"""
    solutions = []
    kaggle_dir = Path(__file__).parent

    for root, dirs, files in os.walk(kaggle_dir):
        if 'solution.py' in files:
            solution_path = Path(root) / 'solution.py'
            solutions.append(solution_path)

    return sorted(solutions)


def run_solution(solution_path):
    """運行單個解決方案"""
    print(f"\n{'='*60}")
    print(f"運行: {solution_path.parent.name}")
    print('='*60)

    try:
        result = subprocess.run(
            [sys.executable, str(solution_path)],
            cwd=solution_path.parent,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"錯誤: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("超時！")
        return False
    except Exception as e:
        print(f"執行失敗: {e}")
        return False


def main():
    """主函數"""
    print("="*60)
    print("Kaggle 解決方案批量運行器")
    print("="*60)

    solutions = find_all_solutions()
    print(f"\n找到 {len(solutions)} 個解決方案\n")

    # 詢問是否要運行所有
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        run_all = True
    else:
        response = input("是否運行所有解決方案？(y/n) [預設: n]: ").strip().lower()
        run_all = response == 'y'

    if run_all:
        success_count = 0
        for solution in solutions:
            if run_solution(solution):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"完成！成功: {success_count}/{len(solutions)}")
        print('='*60)
    else:
        # 列出所有解決方案
        print("\n可用的解決方案:")
        for i, solution in enumerate(solutions, 1):
            print(f"{i:2d}. {solution.parent.name}")

        print("\n提示:")
        print("- 運行所有: python run_all.py --all")
        print("- 運行單個: cd <solution_dir> && python solution.py")


if __name__ == "__main__":
    main()
