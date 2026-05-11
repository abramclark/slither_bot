// ==UserScript==
// @name         Slither.io tasty bot
// @namespace    http://tampermonkey.net/
// @version      2026-03-26
// @description  get thick
// @author       You
// @match        http://slither.io/*
// @grant        none
// ==/UserScript==
$ = q => document.querySelector(q)

function post(path, data){
    const xhr = new XMLHttpRequest()
    xhr.open("POST", "http://localhost:9001/" + path)
    xhr.setRequestHeader("Content-Type", "application/json")
    xhr.onload = e => console.log(e.target.status, e.target.response)
    xhr.send(JSON.stringify(data))
}

window.bot = {
    active: true,
    dead: true,
    toggle: ()=>{
        window.bot.active = !window.bot.active
    },
    interval: null,
    behavior: 1,
    counter: 0,
    t0: 0,

    boost_threshold: 10,
    avoid_dist: 250,
    last_boost: null,

    ws_server: 'ws://localhost:9002',
    ws: null,
    session: null,
}

const _startShowGame = window.startShowGame
window.startShowGame = function() {
    _startShowGame.apply(this, arguments)
    bot.dead = false
    console.log("GAME BEGIN");
    bgp2 = null // disable background for Canvas
    if(bgee) bgee.visible = false // disable BG for WebGL mode
};

function onGameStart() {
    // your bot init here
}

bot.connect = ()=>{
    const url = bot.session ? bot.ws_server + '/' + bot.session : bot.ws_server
    console.log('Connecting to inference server ' + url)
    bot.ws = new WebSocket(url);
    bot.ws.onerror = e => console.error('[bot] ws error', e)
    bot.ws.onclose = e =>{
        console.log('[bot] ws closed code=' + e.code + ' reason=' + e.reason + ' clean=' + e.wasClean, e)
        if(bot.behavior == 1 && bot.active)
            setTimeout(bot.connect, 1000);
    }

    bot.ws.onmessage = e => {
        const [angle, boost, timestamp] = JSON.parse(e.data)
        window.xm = Math.cos(angle) * 100
        window.ym = Math.sin(angle) * 100
        if(boost !== bot.last_boost) { setAcceleration(boost); bot.last_boost = boost }
        bot.counter += 1
        if(!(bot.counter % 10)) {
            const now = new Date().getTime()
            console.log(angle, boost, new Date().getTime() - timestamp)
        }
    }
}
bot.connect()

bot.post_mortem = ()=>{
    if(bot.dead || bot.behavior != 1) return
    console.log('UH OH')
    bot.dead = true
    if(bot.ws.readyState === WebSocket.OPEN) bot.ws.send('[]')
    // wait for game server socket to close before restarting
    if(bot.active) setTimeout(() => { window.want_play = 1 }, 1000)
}

bot.get_record = me =>{
    now = new Date().getTime()
    const relc = (x, y)=> [x - me.xx, y - me.yy]

    const get_props = s =>{
        const meta = [s.wang, s.ang, s.tl, s.sc, s.sp / 14, s.ssp, s.id]
        const segs = [relc(s.xx, s.yy)]

        const skip = Math.max(1, Math.floor(s.pts.length / 30))
        for(let i = 0; i < s.pts.length; i += skip)
            segs.push(relc(s.pts[i].xx, s.pts[i].yy))
        return [meta, segs]
    }

    const food_dat = []
    const fs = foods.filter(f =>{
        if(!f) return
        const [x, y] = relc(f.xx, f.yy)
        const d = Math.sqrt(x * x + y * y)
        food_dat.push([f.sz / 13, x, y])
    })

    me_props = get_props(me)
    const world_ang = Math.atan2(me.yy - window.grd, me.xx - window.grd)
    const edge_x = Math.cos(world_ang) * window.flux_grd + window.grd
    const edge_y = Math.sin(world_ang) * window.flux_grd + window.grd
    const edge_xd = me.xx - edge_x, edge_yd = me.yy - edge_y
    me_props[1][0] = [edge_xd, edge_yd]

    return [me_props, slithers.filter(s => s != me).map(get_props), food_dat, now]
}

bot.behaviors = [

me =>{
    let targetX = 0
    let targetY = 0

    let minDist = Infinity;
    (window.foods || []).forEach(food => {
        if(!food) return

        const dx = food.xx - me.xx
        const dy = food.yy - me.yy
        const dist = Math.sqrt(dx * dx + dy * dy)
        const dist_v = dist / food.sz
        // min dist of 50 * snake_scale prevents endlessly spinning around food
        if (dist_v < minDist && dist > 50 * me.sc) {
            minDist = dist_v;
            targetX = dx
            targetY = dy
        }
    });

    let avoid = null, avoid_segment = null
    minDist = Infinity
    window.slithers.forEach(enemy =>{
        if(enemy == me) return

        enemy.gptz.forEach((p, i) =>{
            const dx = p.xx - me.xx, dy = p.yy - me.yy
            const dist = Math.sqrt(dx * dx + dy * dy)

            if(dist < bot.avoid_dist && dist < minDist){
                avoid = enemy
                minDist = dist
                avoid_segment = i
            }
        })
    })

    if(avoid){
        // calculate avoidance angle
        const avoid_a = Math.atan2(
            avoid.gptz[avoid_segment].yy - me.yy, 
            avoid.gptz[avoid_segment].xx - me.xx,
        ), avoid_asub = angleSub(me.ang, avoid_a)

        if(avoid_asub < Math.PI/2){
            target_a = avoid_a + Math.PI

            targetX = Math.cos(target_a) * 100
            targetY = Math.sin(target_a) * 100
            setAcceleration(0)
        } else setAcceleration(1)
    } else setAcceleration(0)

    window.xm = targetX
    window.ym = targetY
},

me =>{
    const game_state = bot.get_record(me)
    if(bot.ws.readyState === WebSocket.OPEN) bot.ws.send(JSON.stringify(game_state))
},

]

window.onkeydown = ev =>{
    if(ev.which == 66 /* B */) bot.toggle()
    if(ev.which == 67 /* C */) bot.behavior = (bot.behavior + 1) % bot.behaviors.length
}

window.clearInterval(bot.interval); bot.interval = setInterval(()=>{
    const me = window.slither
    if(!bot.active) return
    if(!me || me.dead) {
        bot.post_mortem()
        if(!$('#nick').value) $('#nick').value = 'tasty (bot)'
        return
    }

    bot.behaviors[bot.behavior](me)
}, 200);

function angleSub(from, to) {
    let d = (to - from) % (Math.PI * 2);
    if (d > Math.PI) d -= Math.PI * 2;
    if (d < -Math.PI) d += Math.PI * 2;
    return d;
}
