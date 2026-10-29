"""Headless visual smoke check; does not import LeRobot or touch USB."""
from pathlib import Path
from playwright.sync_api import sync_playwright
from follower_viewer import FollowerViewer


def main():
    viewer = FollowerViewer(0, False)
    errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda message: print("Browser:", message.text) if message.type == "error" else None)
            page.on("requestfailed", lambda request: print("Failed request:", request.url, request.failure))
            viewer.publish({"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0,
                            "wrist_flex": 0, "wrist_roll": 0, "gripper": 20}, "Simulation: pose A", "simulation")
            page.goto(viewer.url)
            try:
                page.wait_for_function("document.querySelector('#positions').rows.length === 6", timeout=15000)
            except Exception:
                print("Page errors:", errors)
                print("Model error:", page.locator("#error").inner_text())
                page.screenshot(path="logs/viewer-failure.png")
                raise
            page.wait_for_timeout(500)
            first = page.locator("canvas").screenshot()
            viewer.publish({"shoulder_pan": 35, "shoulder_lift": -20, "elbow_flex": 30,
                            "wrist_flex": 10, "wrist_roll": 40, "gripper": 80}, "Simulation: pose B", "simulation")
            page.wait_for_function("document.querySelector('#positions').textContent.includes('35.0')")
            page.wait_for_timeout(300)
            second = page.locator("canvas").screenshot()
            assert first != second, "Canvas did not change with joint positions"
            assert not errors, errors
            assert not page.locator("#error").inner_text()
            Path("logs").mkdir(exist_ok=True)
            page.screenshot(path="logs/viewer-preview.png")
            page.wait_for_timeout(1200)
            assert "업데이트 대기" in page.locator("#status").inner_text()
            browser.close()
        print("PASS: 3D model loaded; all 6 readings displayed; poses change; stale data indicated; no JS errors.")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
