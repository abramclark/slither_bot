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
    dead: false,
    toggle: ()=>{
        window.bot.active = !window.bot.active
    },
    interval: null,
    behavior: 1,
    counter: 0,
    t0: 0,

    boost_threshold: 10,
    avoid_dist: 300,

    ws_server: 'ws://localhost:9002',
    ws: null,
}

const _startShowGame = window.startShowGame
window.startShowGame = function() {
    _startShowGame.apply(this, arguments)
    bot.dead = false
};

function onGameStart() {
    console.log("Game started!");
    // your bot init here
}

bot.connect = ()=>{
    console.log('Connecting to inference server ' + bot.ws_server)
    bot.ws = new WebSocket(bot.ws_server);
    bot.ws.onclose = () =>{
        if(bot.behavior == 1 && bot.active)
            setTimeout(bot.connect, 1000);
    }

    bot.ws.onmessage = e => {
        const [norm_angle, boost, time] = JSON.parse(e.data)
        const angle = -norm_angle * Math.PI
        window.xm = Math.cos(angle) * 100
        window.ym = Math.sin(angle) * 100
        setAcceleration(boost)
        bot.counter += 1
        if(!(bot.counter % 10)) {
            const now = new Date().getTime()
            console.log(norm_angle, angle, boost, slither.ang, slither.wang, now - bot.t0)
        }
    }
}
bot.connect()

bot.post_mortem = ()=>{
    if(bot.active) window.want_play = 1
    if(bot.dead || bot.behavior != 1) return
    bot.ws.send('[]')
    bot.dead = true
}

bot.get_record = me =>{
    const norm_angle = a =>{
        a = a / Math.PI
        return a < 1 ? a : -(1 - a % 1)
    }

    const rel_polar = (x, y)=>{
        const dx = x - me.xx, dy = y - me.yy
        return [Math.atan2(dy, dx) / Math.PI, Math.sqrt(dx * dx + dy * dy)]
    }

    const get_props = s =>{
        var data = [norm_angle(s.wang), norm_angle(s.ang), +(s.sp > 5.8), s.sc]
        data = data.concat(rel_polar(s.xx, s.yy))

        const skip = Math.max(1, Math.floor(s.pts.length / 30))
        for(let i = 0; i < s.pts.length; i += skip)
            data = data.concat(rel_polar(s.pts[i].xx, s.pts[i].yy))
        return data
    }

    const food_dat = []
    const fs = foods.filter(f =>{
        if(!f) return
        const [a, d] = rel_polar(f.xx, f.yy)
        if(d < 650) food_dat.push([f.sz, a, d])
    })

    me_props = get_props(me)
    const world_ang = Math.atan2(me.yy - window.grd, me.xx - window.grd)
    const edge_x = Math.cos(world_ang) * window.flux_grd + window.grd
    const edge_y = Math.sin(world_ang) * window.flux_grd + window.grd
    const edge_xd = me.xx - edge_x, edge_yd = me.yy - edge_y
    me_props[4] = norm_angle(world_ang)
    me_props[5] = Math.sqrt(edge_xd * edge_xd + edge_yd * edge_yd)

    return [me.sct + me.fam, food_dat, me_props, slithers.filter(s => s != me).map(get_props)]
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
    bot.t0 = new Date().getTime()
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
