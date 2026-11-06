"""Headless visual check with synthetic encoder frames; no LeRobot or USB."""
from pathlib import Path

from playwright.sync_api import sync_playwright

from follower_feedback import JOINTS
from follower_viewer import FollowerViewer


def main():
    Path('logs').mkdir(exist_ok=True)
    viewer = FollowerViewer(port=0,open_browser=False)
    errors = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='msedge',headless=True)
            page = browser.new_page(viewport={'width':1440,'height':1000})
            page.on('pageerror',lambda error:errors.append(str(error)))
            viewer.publish(dict.fromkeys(JOINTS,0.),'Fixture A',source='simulation',
                           command_positions=dict.fromkeys(JOINTS,2.))
            page.goto(viewer.url)
            page.wait_for_function("document.querySelector('#positions').rows.length === 6")
            page.wait_for_function("document.querySelector('#positions').rows[0].cells[1].textContent === '0.00'")
            assert page.locator('#scene canvas').count()==1
            before = page.locator('#scene canvas').screenshot()
            assert page.locator('#payload-status').count()==0
            assert '100%' not in page.locator('#feedback-status').inner_text()
            viewer.publish(dict.fromkeys(JOINTS,15.),'Fixture B',source='simulation',
                           diagnostics={name:{'voltage_v':12.,'temperature_c':25,'load_raw':2,'current_raw':1,'status':0} for name in JOINTS})
            page.wait_for_function("document.querySelector('#positions').rows[0].cells[1].textContent === '15.00'")
            after = page.locator('#scene canvas').screenshot()
            assert before != after,'Encoder pose changed but rendered robot did not change'
            assert page.locator('#electrical tbody tr').count()==6
            page.wait_for_function("document.querySelector('#status').textContent.includes('업데이트 대기')")
            assert not errors,errors
            page.screenshot(path='logs/viewer-verified.png')
            browser.close()
        print('PASS: local model, measured positions, targets, diagnostics and stale state; no JS errors. No hardware used.')
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
