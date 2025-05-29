import tkinter as tk
import time


class FloatingStopwatch:
    def __init__(self, root):
        self.root = root
        self.root.title("悬浮秒表")
        self.root.overrideredirect(True)  # 移除窗口边框
        self.root.attributes('-topmost', True)  # 保持窗口在最前
        self.root.geometry("200x100+50+50")  # 初始位置和大小

        # 半透明背景
        self.root.attributes('-alpha', 0.8)

        # 拖拽功能变量
        self._offsetx = 0
        self._offsety = 0

        # 计时器变量
        self.running = False
        self.start_time = 0
        self.elapsed_time = 0

        # 创建UI
        self.create_widgets()

        # 绑定鼠标事件
        self.bind_events()

    def create_widgets(self):
        # 时间显示标签
        self.time_label = tk.Label(
            self.root,
            text="点击开始计时",
            font=('楷体', 20),
            bg='black',
            fg='white'
        )
        self.time_label.pack(fill=tk.BOTH, expand=True)

        # 控制按钮框架
        button_frame = tk.Frame(self.root, bg='black')
        button_frame.pack(fill=tk.X)

        # 开始/暂停按钮
        self.start_button = tk.Button(
            button_frame,
            text="开始",
            command=self.toggle_start_stop,
            bg='#333333',
            fg='white',
            relief=tk.FLAT
        )
        self.start_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # 重置按钮
        reset_button = tk.Button(
            button_frame,
            text="重置",
            command=self.reset,
            bg='#333333',
            fg='white',
            relief=tk.FLAT
        )
        reset_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # 关闭按钮
        close_button = tk.Button(
            button_frame,
            text="关闭",
            command=self.root.destroy,
            bg='#333333',
            fg='white',
            relief=tk.FLAT
        )
        close_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

    def bind_events(self):
        # 绑定拖动事件
        self.time_label.bind("<Button-1>", self.clickwin)
        self.time_label.bind("<B1-Motion>", self.dragwin)

        # 右键菜单
        self.root.bind("<Button-3>", self.show_context_menu)

    def clickwin(self, event):
        # 记录鼠标点击位置
        self._offsetx = event.x
        self._offsety = event.y

    def dragwin(self, event):
        # 移动窗口
        x = self.root.winfo_pointerx() - self._offsetx
        y = self.root.winfo_pointery() - self._offsety
        self.root.geometry(f"+{x}+{y}")

    def show_context_menu(self, event):
        # 创建右键菜单
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="置顶", command=self.toggle_topmost)
        menu.add_command(label="透明度+", command=lambda: self.change_opacity(0.1))
        menu.add_command(label="透明度-", command=lambda: self.change_opacity(-0.1))
        menu.add_separator()
        menu.add_command(label="退出", command=self.root.destroy)

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def toggle_topmost(self):
        # 切换置顶状态
        current = self.root.attributes('-topmost')
        self.root.attributes('-topmost', not current)

    def change_opacity(self, delta):
        # 调整透明度
        current = self.root.attributes('-alpha')
        new = max(0.1, min(1.0, current + delta))
        self.root.attributes('-alpha', new)

    def toggle_start_stop(self):
        # 开始/暂停计时
        if not self.running:
            self.start()
        else:
            self.stop()

    def start(self):
        # 开始计时
        if not self.running:
            self.running = True
            self.start_time = time.time() - self.elapsed_time
            self.start_button.config(text="暂停")
            self.update_time()

    def stop(self):
        # 暂停计时
        if self.running:
            self.running = False
            self.elapsed_time = time.time() - self.start_time
            self.start_button.config(text="开始")

    def reset(self):
        # 重置计时器
        self.running = False
        self.elapsed_time = 0
        self.time_label.config(text="00:00:00.000")
        self.start_button.config(text="开始")

    def update_time(self):
        # 更新显示的时间
        if self.running:
            current_time = time.time() - self.start_time
            self.elapsed_time = current_time

            # 格式化时间: 小时:分钟:秒.毫秒
            hours = int(current_time // 3600)
            minutes = int((current_time % 3600) // 60)
            seconds = int(current_time % 60)
            milliseconds = int((current_time - int(current_time)) * 1000)

            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
            self.time_label.config(text=time_str)

            # 每10毫秒更新一次
            self.root.after(10, self.update_time)


if __name__ == "__main__":
    root = tk.Tk()
    app = FloatingStopwatch(root)
    root.mainloop()