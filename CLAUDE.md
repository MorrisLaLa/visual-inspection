# 專案筆記（給 Claude 的記憶）

## 📲 讓 Google Apps Script 網頁在手機主畫面有自訂圖示（已驗證可用）

### 結論：用 GitHub Pages 做一個「圖示＋跳轉」小網頁
Apps Script 網頁（script.google.com/.../exec）在 Google 的網域底下，**無法自訂主畫面圖示**
（Index.html 在 iframe 裡，apple-touch-icon 無效）。解法是另做一個靜態頁：

1. 在 repo 建資料夾（例：`pig/`），放：
   - `apple-touch-icon.png`（180×180，iPhone 用）
   - `icon-512.png`（Android / favicon 用，`<link rel="icon">`）
   - `index.html`：
     - `<title>` 和 `<meta name="apple-mobile-web-app-title">` 設成主畫面名稱
     - `<link rel="apple-touch-icon" href="apple-touch-icon.png">`
     - 頁面顯示「分享 → 加入主畫面 → 新增」3 步驟 + 「直接進入」按鈕
     - JS：載入後 `history.replaceState(..., '?app=1')`，所以「加入主畫面」存到的是 `?app=1` 網址；
       網址帶 `?app=1` 時 `location.replace(Apps Script 網址)` → 從圖示打開就直接跳進系統
     - **不要**設 `apple-mobile-web-app-capable`（全螢幕模式跳到 Google 網域會變成另一個瀏覽框、登入狀態可能不同）
2. 使用者在 GitHub：Settings → Pages → Branch 選放檔案的分支、`/(root)` → Save（這一步 Claude 無權限，要使用者按）。
3. 網址：`https://<帳號>.github.io/<repo>/<資料夾>/`。iPhone 用 **Safari** 打開 → 分享 → 加入主畫面。
4. App 內的「📲 加到主畫面」按鈕直接連到這個頁面：LINE 內 iPhone 用 `x-safari-https://…`，Android 用 `intent://…;package=com.android.chrome;…`。

### 試過、不要再用的方法（使用者覺得非常麻煩）
- **iOS 描述檔 .mobileconfig（Web Clip）**：從 Gmail App、雲端硬碟 App、Chrome、Google App 打開都只會顯示 XML 程式碼；
  Safari 從雲端硬碟下載會先跳「無法掃毒」，按「仍要下載」也只存到下載項目，不會跳安裝。
  只有 Safari（正確 MIME）、內建「郵件」App、AirDrop 才能裝。→ 不要再推這個方法。
- **raw.githack / rawcdn.githack**：瀏覽器會先出現「External Content Notice」插頁，不能用。
- **jsDelivr / statically**：HTML 會被當純文字，不能用。
- **「捷徑」App 自訂圖示**：可行但步驟多，使用者不想做。

### 現況（PIG 追蹤）
- 圖示頁：`pig/`（GitHub Pages 從分支 `claude/loving-ritchie-6npoqe` 發布）→ https://morrislala.github.io/visual-inspection/pig/
- **不要刪這個分支、不要關 Pages**，否則主畫面圖示會打不開。
- 圖示設計：白底、Inter Display Black 字「P」紅 #C00000、「IG」藍 #0A5FA8，下方一條追蹤線（紅色已走段＋紅色膠囊＝PIG、藍色未走段、站點圓圈），
  跟使用者「一六工程日誌」的 UNILUX 紅藍配色同系列。

## 使用者偏好
- 溝通用繁體中文；要直接動手、少問問題、少丟長步驟。
- **程式碼（Code.gs / Index.html 等）不要放 GitHub**，用私人一鍵複製頁交付；只有不含程式的靜態頁（如圖示頁）經同意後可放。
