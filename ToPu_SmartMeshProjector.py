bl_info = {
    "name": "ToPu_SmartMeshProjector",
    "author": "http4211",
    "version": (1, 2),
    "blender": (4, 0, 0),
    "location": "3D View > 編集モードの右クリックメニュー",
    'tracker_url': 'https://github.com/http4211/ToPu_SmartMeshProjector',
    "description": "メッシュを選択したメッシュに投影",
    "category": "3D View"
}

import bpy
import bmesh
import gpu
import blf
import math
import functools
import inspect
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix, Euler
from mathutils.bvhtree import BVHTree
from bpy_extras import view3d_utils
from bpy.props import IntVectorProperty


# === グローバル一時保持 ===
highlight_locked = False
cached_edit_objects = []
cached_edit_coords = {}  # 各オブジェクト名に対応する座標リスト
cached_edit_verts_indices = {}  # 各オブジェクト名に対応する頂点インデックス
cached_normal_matrix = None
gizmo_handler = None
gizmo_showing = False
hover_axis = None
selected_axis = None
mouse_tracker_running = False
highlight_handle = None
vertex_batch = None
edge_batch = None
face_batch = None
help_text_handle = None
temp_gizmo_orientation_matrix = None



# === SnapDrawState: スナップ対象の状態を一時保持する ===
class SnapDrawState:
    def __init__(self):
        self.coords = []
        self.mode = 'POINTS'
        self.dirty = False
        self.target_obj = None
        self.target_face_index = None
        self.snap_location = None  # 決定スナップ座標

    def set(self, coords, mode='POINTS', obj=None, face_index=None, snap_location=None):
        self.coords = coords
        self.mode = mode
        self.dirty = True
        self.target_obj = obj
        self.target_face_index = face_index
        self.snap_location = snap_location

    def clear(self):
        self.coords = []
        self.dirty = True
        self.target_obj = None
        self.target_face_index = None
        self.snap_location = None

    def draw_callback(self):
        #print(f"[DRAW] Drawing snap highlight: {len(self.coords)} points in mode {self.mode}")
        if not self.coords:
            return

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')  # ← 3D用に変更
        shader.bind()

        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')  # ← これを追加！ 奥行きテストを無効化
        gpu.state.line_width_set(3.0)
        gpu.state.point_size_set(12.0)  # 目立たせる

        #print(f"[DRAW] Drawing snap highlight: {len(self.coords)} points in mode {self.mode}")

        if self.mode == 'POINTS':
            shader.uniform_float("color", (1.0, 1.0, 0.0, 1.0))
            batch = batch_for_shader(shader, 'POINTS', {"pos": self.coords})
            batch.draw(shader)

        elif self.mode == 'LINES':
            shader.uniform_float("color", (0.0, 0.5, 1.0, 1.0))
            batch = batch_for_shader(shader, 'LINES', {"pos": self.coords})
            batch.draw(shader)

        elif self.mode == 'LINE_STRIP':
            shader.uniform_float("color", (0.0, 1.0, 0.0, 1.0))
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": self.coords})
            batch.draw(shader)

        gpu.state.depth_test_set('LESS_EQUAL')  # ← 元に戻す
        gpu.state.blend_set('NONE')



snap_draw_state = SnapDrawState()
snap_draw_handler = None

def compute_orientation_matrix_from_snap(context):
    #print("compute_orientation_matrix_from_snap (advanced)")
    obj = snap_draw_state.target_obj
    index = snap_draw_state.target_face_index
    coords = snap_draw_state.coords
    mode = snap_draw_state.mode
    location = snap_draw_state.snap_location

    if not obj or index is None or not coords:
        return None

    depsgraph = context.evaluated_depsgraph_get()
    mw = obj.matrix_world.to_3x3()

    if obj.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(obj.data)
    else:
        bm = bmesh.new()
        bm.from_object(obj, depsgraph)

    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    def build_rotation_matrix(z_axis, hint_x_axis):
        temp_x = hint_x_axis
        y_axis = z_axis.cross(temp_x).normalized()
        x_axis = y_axis.cross(z_axis).normalized()
        return Matrix((x_axis, y_axis, z_axis)).transposed()

    if mode == 'LINE_STRIP' and index < len(bm.faces):
        face = bm.faces[index]
        z_axis = (mw @ face.normal).normalized()

        if len(face.verts) == 3:
            world_verts = [obj.matrix_world @ v.co for v in face.verts]
            angles = []
            for i in range(3):
                a = world_verts[i]
                b = world_verts[(i + 1) % 3]
                c = world_verts[(i + 2) % 3]
                ba = (a - b).normalized()
                bc = (c - b).normalized()
                angle = ba.angle(bc)
                angles.append((angle, b))
            angles.sort(key=lambda x: x[0])
            sharp_point = angles[0][1]
            center = sum(world_verts, Vector()) / 3
            hint_x_axis = (center - sharp_point).normalized()
            return build_rotation_matrix(z_axis, hint_x_axis)

        else:
            world_edges = [(obj.matrix_world @ e.verts[0].co, obj.matrix_world @ e.verts[1].co) for e in face.edges]
            center_pairs = []
            for i, (v1a, v1b) in enumerate(world_edges):
                center1 = (v1a + v1b) / 2
                for j, (v2a, v2b) in enumerate(world_edges):
                    if i >= j:
                        continue
                    center2 = (v2a + v2b) / 2
                    vec = (center2 - center1).normalized()
                    angle = abs(vec.dot(z_axis))
                    if angle < 0.2:
                        center_pairs.append((center1, center2))

            if center_pairs:
                c1, c2 = max(center_pairs, key=lambda pair: (pair[1] - pair[0]).length)
                x_dir = (c2 - c1).normalized()
            else:
                longest_edge = max(face.edges, key=lambda e: (e.verts[0].co - e.verts[1].co).length)
                x_dir = (obj.matrix_world @ longest_edge.verts[1].co - obj.matrix_world @ longest_edge.verts[0].co).normalized()

            return build_rotation_matrix(z_axis, x_dir)

    elif mode == 'LINES' and index < len(bm.edges):
        edge = bm.edges[index]
        v1 = obj.matrix_world @ edge.verts[0].co
        v2 = obj.matrix_world @ edge.verts[1].co
        edge_dir = (v2 - v1).normalized()
        face = bm.faces[0] if bm.faces else None
        z_axis = (mw @ face.normal).normalized() if face else Vector((0, 0, 1))
        return build_rotation_matrix(z_axis, edge_dir)

    elif mode == 'POINTS' and index < len(bm.verts):
        vertex = bm.verts[index]
        z_axis = (mw @ vertex.normal).normalized()
        fallback_x = Vector((1, 0, 0))
        if abs(z_axis.dot(fallback_x)) > 0.99:
            fallback_x = Vector((0, 1, 0))
        return build_rotation_matrix(z_axis, fallback_x)

    if obj.mode != 'EDIT':
        bm.free()

    return None



def create_orientation_using_matrix(self, context, matrix, name="TempSnap"):
    import bpy
    from mathutils import Matrix

    # モードを OBJECT に
    bpy.ops.object.mode_set(mode='OBJECT')

    # ダミー Empty を作成
    empty = bpy.data.objects.new("DummyOrientation", None)
    context.collection.objects.link(empty)

    origin = context.scene.cursor.location
    empty.matrix_world = Matrix.Translation(origin) @ matrix.to_4x4()

    # オブジェクトを選択状態にする
    bpy.ops.object.select_all(action='DESELECT')
    empty.select_set(True)
    context.view_layer.objects.active = empty

    # 🔧 override context の明示的な構築
    override = context.copy()
    override["area"] = next((a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
    override["region"] = next((r for r in override["area"].regions if r.type == 'WINDOW'), None)
    override["space_data"] = override["area"].spaces.active
    override["region_data"] = override["space_data"].region_3d
    override["active_object"] = empty
    override["selected_objects"] = [empty]
    override["selected_editable_objects"] = [empty]

    try:
        # 🔧 引数をすべて kwargs にまとめて渡す（Blender 4.x仕様対応）
        bpy.ops.transform.create_orientation(
            override,
            **{
                "name": name,
                "use": True,
                "overwrite": True
            }
        )
        slot = context.scene.transform_orientation_slots[0]
        if slot.custom_orientation:
            slot.custom_orientation.matrix = matrix
            self.custom_orientation_name = name
            #print(f"[INFO] Transform Orientation '{name}' を作成しました")
        else:
            print("[WARN] Transform Orientation は作成されたが取得できませんでした")

    except Exception as e:
        print(f"[ERROR] Transform Orientation 作成失敗: {e}")

    finally:
        bpy.data.objects.remove(empty, do_unlink=True)






def update_snap_highlight(context, event):
    region = context.region
    rv3d = context.region_data

    if not region or not rv3d:
        for area in context.window.screen.areas:
            if area.type == 'VIEW_3D':
                for reg in area.regions:
                    if reg.type == 'WINDOW':
                        region = reg
                        space = area.spaces.active
                        if space.type == 'VIEW_3D':
                            rv3d = space.region_3d
                        break
                break

    if not region or not rv3d:
        snap_draw_state.clear()
        return

    coord = context.window_manager.mouse_pos
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    ray_direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    depsgraph = context.evaluated_depsgraph_get()
    hit, location, normal, face_index, obj, _ = context.scene.ray_cast(
        depsgraph, ray_origin, ray_direction)

    if not obj or obj.type != 'MESH':
        snap_draw_state.clear()
        return

    if obj.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(obj.data)
    else:
        bm = bmesh.new()
        bm.from_object(obj, depsgraph)

    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    best_coords = []
    best_mode = ''
    best_index = None
    best_dist = float('inf')

    # --- 面 ---
    for f in bm.faces:
        verts = [obj.matrix_world @ v.co for v in f.verts]
        if len(verts) >= 3:
            center = sum(verts, Vector()) / len(verts)
            dist = (center - location).length
            if dist < best_dist:
                best_coords = verts + [verts[0]]
                best_mode = 'LINE_STRIP'
                best_index = f.index
                best_dist = dist

    # --- 辺 ---
    for e in bm.edges:
        if len(e.verts) == 2:
            v1 = obj.matrix_world @ e.verts[0].co
            v2 = obj.matrix_world @ e.verts[1].co
            center = (v1 + v2) / 2
            dist = (center - location).length
            if dist < best_dist:
                best_coords = [v1, v2]
                best_mode = 'LINES'
                best_index = e.index
                best_dist = dist

    # --- 頂点 ---
    for v in bm.verts:
        world_co = obj.matrix_world @ v.co
        dist = (world_co - location).length
        if dist < best_dist:
            best_coords = [world_co]
            best_mode = 'POINTS'
            best_index = v.index
            best_dist = dist

    if not best_coords:
        snap_draw_state.clear()
        if obj.mode != 'EDIT':
            bm.free()
        return

    snap_draw_state.set(
        coords=best_coords,
        mode=best_mode,
        obj=obj,
        face_index=best_index,
        snap_location=location
    )

    if obj.mode != 'EDIT':
        bm.free()

    #print(f"[SNAP] mode={best_mode}, index={best_index}, len(coords)={len(best_coords)}")






def safe_tag_redraw(context):
    area = context.area
    if not area or area.type != 'VIEW_3D':
        for a in context.window.screen.areas:
            if a.type == 'VIEW_3D':
                area = a
                break
    if area:
        area.tag_redraw()



class LaunchMeshProjectorOperator(bpy.types.Operator):
    bl_idname = "view3d.launch_mesh_projector"
    bl_label = "メッシュ投影ギズモ起動"

    @classmethod
    def poll(cls, context):
        obj = context.edit_object
        return obj and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        # ★ 遅延呼び出しでモーダルを正しく起動
        def delayed_call():
            bpy.ops.view3d.modal_mesh_projector('INVOKE_DEFAULT')
            return None  # タイマーを止める

        # 0.01秒後にモーダルオペレーターを起動
        bpy.app.timers.register(delayed_call, first_interval=0.01)
        return {'FINISHED'}









# === コンテキストメニューに表示する関数 ===
def draw_mesh_projector_menu(self, context):
    layout = self.layout
    layout.separator()
    layout.operator("view3d.launch_mesh_projector", text="メッシュ投影", icon='MOD_SHRINKWRAP')




def draw_gizmo_help_text():
    font_id = 0
    blf.size(font_id, 16)
    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)  
    
    lines = [
        "操作ガイド",
        "・軸を選択",
        "　・Alt+左クリック：軸をカスタム",
        "　・Alt+右クリック：軸をリセット",
        "　・トランスフォーム座標系により初期軸が設定されます",
        "・投影先を選択",
        "・Space/Enter：投影実行",
        "・ESC / 右クリック：キャンセル",
    ]

    x, y = 60, 50 + len(lines) * 20  # ← 少し右に & 行間広め
    for line in lines:
        blf.position(font_id, x, y, 0)
        blf.draw(font_id, line)
        y -= 20



def update_highlight_batches():
    global vertex_batch, edge_batch, face_batch

    vertex_coords = []
    edge_coords = []
    face_coords = []

    for obj in cached_edit_objects:
        if obj.type != 'MESH':
            continue

        mesh = obj.data
        verts = mesh.vertices
        edges = mesh.edges
        polys = mesh.polygons

        indices = cached_edit_verts_indices.get(obj.name, [])

        vertex_coords.extend([
            tuple(obj.matrix_world @ verts[i].co)
            for i in indices if i < len(verts)
        ])

        for e in edges:
            if e.vertices[0] in indices and e.vertices[1] in indices:
                edge_coords.extend([
                    tuple(obj.matrix_world @ verts[e.vertices[0]].co),
                    tuple(obj.matrix_world @ verts[e.vertices[1]].co),
                ])

        for f in polys:
            if all(i in indices for i in f.vertices) and len(f.vertices) >= 3:
                base = obj.matrix_world @ verts[f.vertices[0]].co
                for i in range(1, len(f.vertices) - 1):
                    v1 = obj.matrix_world @ verts[f.vertices[i]].co
                    v2 = obj.matrix_world @ verts[f.vertices[i + 1]].co
                    face_coords.extend([tuple(base), tuple(v1), tuple(v2)])

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    vertex_batch = batch_for_shader(shader, 'POINTS', {"pos": vertex_coords}) if vertex_coords else None
    edge_batch = batch_for_shader(shader, 'LINES', {"pos": edge_coords}) if edge_coords else None
    face_batch = batch_for_shader(shader, 'TRIS', {"pos": face_coords}) if face_coords else None




def draw_highlight_overlay():
    global vertex_batch, edge_batch, face_batch

    context = bpy.context
    if not cached_edit_objects:
        return

    update_highlight_batches()

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    gpu.state.point_size_set(4.0)

    if vertex_batch:
        shader.uniform_float("color", (0.0, 1.0, 0.0, 0.25))
        vertex_batch.draw(shader)
    if edge_batch:
        shader.uniform_float("color", (0.0, 1.0, 0.0, 0.25))
        edge_batch.draw(shader)
    if face_batch:
        shader.uniform_float("color", (0.0, 1.0, 0.0, 0.1))
        face_batch.draw(shader)

    gpu.state.blend_set('NONE')




# === ユーザー設定を反映した軸行列 ===
def get_orientation_matrix(context, obj):
    orientation = context.scene.transform_orientation_slots[0].type
    cursor = context.scene.cursor

    if orientation == 'GLOBAL':
        return Matrix.Identity(3)

    elif orientation == 'LOCAL':
        return obj.matrix_world.to_3x3()

    elif orientation == 'NORMAL':
        global cached_normal_matrix
        if cached_normal_matrix:
            return obj.matrix_world.to_3x3() @ cached_normal_matrix
        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
            selected_faces = [f for f in bm.faces if f.select]
            if selected_faces:
                return selected_faces[0].normal.to_track_quat('Z', 'Y').to_matrix()
        return obj.matrix_world.to_3x3()

    elif orientation == 'VIEW':
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        rv3d = region.data
                        if rv3d:
                            return rv3d.view_rotation.to_matrix()
        return Matrix.Identity(3)

    elif orientation == 'CURSOR':
        if cursor.rotation_mode == 'QUATERNION':
            return cursor.rotation_quaternion.to_matrix().to_3x3()
        elif cursor.rotation_mode == 'AXIS_ANGLE':
            angle = cursor.rotation_axis_angle[0]
            axis = Vector(cursor.rotation_axis_angle[1:]).normalized()
            return Matrix.Rotation(angle, 3, axis)
        else:
            try:
                return Euler(cursor.rotation_euler, cursor.rotation_mode).to_matrix()
            except Exception as e:
                #print("カーソル回転取得失敗:", e)
                return Matrix.Identity(3)

    elif orientation == 'GIMBAL':
        return obj.matrix_world.to_3x3()  # GIMBALは簡易的にローカルで代用

    else:
        # ✅ オペレーターが内部にカスタム座標系を持っていればそれを使う
        stack = inspect.stack()
        for frame in stack:
            local_self = frame.frame.f_locals.get('self')
            if hasattr(local_self, "temp_orientation_matrix") and local_self.temp_orientation_matrix:
                return local_self.temp_orientation_matrix

        # それがなければ通常のカスタム
        custom = context.scene.transform_orientation_slots[0].custom_orientation
        if custom:
            return custom.matrix

        return Matrix.Identity(3)


# === ピボット位置（ギズモ中心） ===
def get_pivot_location(context, obj):
    global cached_edit_coords

    if cached_edit_coords and obj.name in cached_edit_coords:
        coords = cached_edit_coords[obj.name]
        if coords:
            avg = sum(coords, Vector()) / len(coords)
            return obj.matrix_world @ avg

    return obj.matrix_world.translation



def calculate_constant_length_3d(context, region, rv3d, origin, pixel_length=80):
    """
    ビュー座標で pixel_length ピクセルに相当する 3D 長さを返す
    """
    from bpy_extras import view3d_utils
    origin_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, origin)
    if not origin_2d:
        return 1.0  # fallback
    offset_2d = origin_2d + Vector((pixel_length, 0))
    offset_3d = view3d_utils.region_2d_to_location_3d(region, rv3d, offset_2d, origin)
    return (offset_3d - origin).length

def draw_cylinder_axis(shader, start, end, color, radius=0.02, segments=16):
    direction = (end - start).normalized()
    length = (end - start).length

    if direction.length == 0:
        return

    # 基本ベクトル（正規直交基底）
    side = direction.orthogonal().normalized()
    up = direction.cross(side).normalized()

    # 円柱の断面円の点を生成
    top_center = end
    bottom_center = start
    verts_top = []
    verts_bottom = []

    for i in range(segments):
        angle = 2 * math.pi * i / segments
        offset = side * math.cos(angle) * radius + up * math.sin(angle) * radius
        verts_top.append(top_center + offset)
        verts_bottom.append(bottom_center + offset)

    # 側面の三角形群
    coords = []
    for i in range(segments):
        a1 = verts_bottom[i]
        a2 = verts_bottom[(i + 1) % segments]
        b1 = verts_top[i]
        b2 = verts_top[(i + 1) % segments]
        coords.extend([a1, b1, a2])
        coords.extend([a2, b1, b2])

    batch = batch_for_shader(shader, 'TRIS', {"pos": coords})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def draw_arrowhead(shader, origin, end, color, size=0.25, segments=16, offset_ratio=0.15):
    direction = (end - origin).normalized()
    if direction.length == 0:
        return

    side = direction.orthogonal().normalized()
    up = direction.cross(side).normalized()

    # 軸の先からさらに offset 分だけ先に tip を移動
    offset = (end - origin).length * offset_ratio
    tip = end + direction * offset
    base_center = tip - direction * size
    radius = size * 0.4

    # ベース円周の頂点群を作成
    circle_verts = []
    for i in range(segments):
        angle = (2 * math.pi * i) / segments
        offset_vec = side * math.cos(angle) * radius + up * math.sin(angle) * radius
        circle_verts.append(base_center + offset_vec)

    # 円錐サーフェス：tip → 周辺 → base_center
    coords = []
    for v in circle_verts:
        coords.extend([tip, v, base_center])

    batch = batch_for_shader(shader, 'TRIS', {"pos": coords})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)



def check_axis_hover(context, event, region=None, rv3d=None):
    global hover_axis

    # 明示的に region/rv3d を補完（右クリック起動でも動くように）
    if not region or not rv3d:
        for area in context.window.screen.areas:
            if area.type == 'VIEW_3D':
                for reg in area.regions:
                    if reg.type == 'WINDOW':
                        region = reg
                        space = area.spaces.active
                        if space.type == 'VIEW_3D':
                            rv3d = space.region_3d
                        break
                break

    if not region or not rv3d:
        hover_axis = None
        return


    obj = cached_edit_objects if cached_edit_objects else context.active_object
    if not obj:
        hover_axis = None
        return

    obj = cached_edit_objects[0] if cached_edit_objects else context.active_object
    origin = get_pivot_location(context, obj)
    scale = calculate_constant_length_3d(context, region, rv3d, origin, pixel_length=80)
    basis = temp_gizmo_orientation_matrix if temp_gizmo_orientation_matrix else get_orientation_matrix(context, obj)


    axes = {
        "X": basis @ Vector((1, 0, 0)) * scale,
        "Y": basis @ Vector((0, 1, 0)) * scale,
        "Z": basis @ Vector((0, 0, 1)) * scale,
    }

    mouse_pos = Vector(context.window_manager.mouse_pos)
    min_dist = 12  # ピクセル距離でのしきい値
    found = None

    for axis, vec in axes.items():
        start_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, origin)
        end_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, origin + vec)
        #print(f"[DEBUG] axis={axis} start_2d={start_2d} end_2d={end_2d}")
        if not (start_2d and end_2d):
            #print(f"[WARN] axis {axis}: start_2d or end_2d is None — skipping")
            continue

        line_vec = end_2d - start_2d
        line_dir = line_vec.normalized()
        proj_len = (mouse_pos - start_2d).dot(line_dir)
        proj_point = start_2d + line_dir * proj_len
        dist = (mouse_pos - proj_point).length
        #print(f"[HOVER] axis={axis} dist={dist:.2f}, proj_len={proj_len:.2f}, threshold={min_dist}")

        if 0 <= proj_len <= line_vec.length and dist < min_dist:
            #print(f"[INFO] hover match: axis={axis} dist={dist:.2f}")
            min_dist = dist
            found = axis

    hover_axis = found
    #print(f"[RESULT] hover_axis = {hover_axis}")



# === ギズモ描画 ===
def draw_translate_gizmo():
    global hover_axis
    context = bpy.context
    obj = cached_edit_objects[0] if cached_edit_objects else context.active_object
    if not obj:
        return

    region = context.region
    rv3d = context.region_data

    # region / rv3d がないときは自前で探す（右クリック起動対応）
    if not region or not rv3d:
        for area in context.window.screen.areas:
            if area.type == 'VIEW_3D':
                for reg in area.regions:
                    if reg.type == 'WINDOW':
                        region = reg
                        space = area.spaces.active
                        if space.type == 'VIEW_3D':
                            rv3d = space.region_3d
                        break
                break

    if not region or not rv3d:
        return


    obj = cached_edit_objects[0] if cached_edit_objects else context.active_object
    origin = get_pivot_location(context, obj)

    # ✅ ビュー距離に応じたスケール：ピクセル80に相当する3D長さ
    scale_factor = calculate_constant_length_3d(context, region, rv3d, origin, pixel_length=50)

    basis = temp_gizmo_orientation_matrix if temp_gizmo_orientation_matrix else get_orientation_matrix(context, obj)

    axes = {
        "X": (origin, origin + basis @ Vector((1, 0, 0)) * scale_factor),
        "Y": (origin, origin + basis @ Vector((0, 1, 0)) * scale_factor),
        "Z": (origin, origin + basis @ Vector((0, 0, 1)) * scale_factor),
    }

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.line_width_set(12.0)

    colors = {
        "X": (1.0, 0.2, 0.2, 0.5),
        "Y": (0.2, 1.0, 0.2, 0.5),
        "Z": (0.4, 0.6, 1.0, 0.5),
    }

    for axis, (start, end) in axes.items():
        if axis == selected_axis:
            color = (1.0, 1.0, 1.0, 1.0)  # 真っ白で明確に
        elif axis == hover_axis:
            color = (1.0, 1.0, 0.3, 1.0)  # 黄色でハイライト
        else:
            color = colors[axis]  # 通常の赤／緑／青

        shader.bind()
        shader.uniform_float("color", color)
        
        offset_ratio = 0.12  # 軸の20%を根元オフセットする
        adjusted_start = start.lerp(end, offset_ratio)
        draw_cylinder_axis(shader, adjusted_start, end, color, radius=scale_factor * 0.04)
        
        #draw_arrowhead(shader, start, end, color, size=scale_factor * 0.2)






# === モーダル起動時にピボット座標と投影対象インデックスをキャッシュ ===
class ModalMeshProjectorOperator(bpy.types.Operator):
    bl_idname = "view3d.modal_mesh_projector"
    bl_label = "Modal Mesh Projector"

    @classmethod
    def poll(cls, context):
        obj = context.edit_object
        return obj and obj.type == 'MESH' and obj.mode == 'EDIT'


    def invoke(self, context, event=None):
        global gizmo_handler, gizmo_showing, mouse_tracker_running
        global cached_edit_objects, cached_edit_coords, cached_edit_verts_indices
        global cached_normal_matrix
        global highlight_handle, help_text_handle
        global temp_gizmo_orientation_matrix
        
        self.custom_orientation_name = ""
        self.temp_orientation_matrix = None

        self.prev_orientation = context.scene.transform_orientation_slots[0].type

        area = context.area
        if not area or area.type != 'VIEW_3D':
            for a in context.window.screen.areas:
                if a.type == 'VIEW_3D':
                    area = a
                    break
            if not area:
                self.report({'ERROR'}, "3Dビューが見つかりません")
                return {'CANCELLED'}

        region = None
        for r in area.regions:
            if r.type == 'WINDOW':
                region = r
                break
        if not region:
            self.report({'ERROR'}, "3Dビューのリージョンが見つかりません")
            return {'CANCELLED'}

        # 描画ハンドラ登録
        if not highlight_handle:
            highlight_handle = bpy.types.SpaceView3D.draw_handler_add(draw_highlight_overlay, (), 'WINDOW', 'POST_VIEW')
        if not gizmo_handler:
            gizmo_handler = bpy.types.SpaceView3D.draw_handler_add(draw_translate_gizmo, (), 'WINDOW', 'POST_VIEW')
        if not help_text_handle:
            help_text_handle = bpy.types.SpaceView3D.draw_handler_add(draw_gizmo_help_text, (), 'WINDOW', 'POST_PIXEL')

        # 編集オブジェクト取得（ここが最重要！）
        cached_edit_objects = [
            o for o in context.view_layer.objects
            if o.select_get() and o.mode == 'EDIT' and o.type == 'MESH'
        ]

        if not cached_edit_objects:
            self.report({'WARNING'}, "編集モード中のメッシュが選択されていません")
            return {'CANCELLED'}

        cached_edit_coords = {}
        cached_edit_verts_indices = {}

        for obj in cached_edit_objects:
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            cached_edit_coords[obj.name] = [v.co.copy() for v in bm.verts if v.select]
            cached_edit_verts_indices[obj.name] = [i for i, v in enumerate(bm.verts) if v.select]

        cached_normal_matrix = None

        update_highlight_batches()

        # ギズモの回転軸を初期化
        obj = context.active_object
        if obj:
            temp_gizmo_orientation_matrix = get_orientation_matrix(context, cached_edit_objects[0]) if cached_edit_objects else Matrix.Identity(3)

        gizmo_showing = True
        context.window_manager.modal_handler_add(self)

        register_snap_draw_handler(context.area)

        def trigger_tracker():
            try:
                bpy.ops.wm.modal_mouse_tracker('INVOKE_DEFAULT')
            except Exception as e:
                print("トラッカー起動エラー:", e)
            return None

        bpy.app.timers.register(trigger_tracker, first_interval=0.01)

        return {'RUNNING_MODAL'}





    # === モーダル処理の軸選択とハイライト固定化対応 + オブジェクト選択許可 ===
    def modal(self, context, event):
        global selected_axis, gizmo_showing, gizmo_handler, highlight_handle, highlight_locked
        global help_text_handle, temp_gizmo_orientation_matrix 

        # ✅ altキーが押されている間、スナップハイライト更新
        if event.alt:
            #print("[DEBUG] alt押下中、スナップハイライト更新")
            update_snap_highlight(context, event)
            safe_tag_redraw(context)

            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            overlay = space.overlay
                            # 初回押下時のみ、元の状態を記録
                            if not hasattr(self, "_wire_overlay_prev"):
                                self._wire_overlay_prev = overlay.show_wireframes
                                self._wire_threshold_prev = overlay.wireframe_threshold
                                self._wire_opacity_prev = overlay.wireframe_opacity
                            overlay.show_wireframes = True
                            overlay.wireframe_threshold = 1
                            overlay.wireframe_opacity = 0.5

        else:
            # altを離したらハイライト解除
            if snap_draw_state.coords:
                #print("[DEBUG] alt解除 → ハイライト非表示")
                snap_draw_state.clear()
                safe_tag_redraw(context)
                
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D' and hasattr(self, "_wire_overlay_prev"):
                            overlay = space.overlay
                            overlay.show_wireframes = self._wire_overlay_prev
                            overlay.wireframe_threshold = self._wire_threshold_prev
                            overlay.wireframe_opacity = self._wire_opacity_prev
                            del self._wire_overlay_prev
                            del self._wire_threshold_prev
                            del self._wire_opacity_prev

            
        # ✅ マウス移動 → 座標補正＋軸ホバー判定
        if event.type == 'MOUSEMOVE':
            for area in context.window.screen.areas:
                if area.type == 'VIEW_3D':
                    for reg in area.regions:
                        if reg.type == 'WINDOW':
                            region = reg
                            space = area.spaces.active
                            if space.type == 'VIEW_3D':
                                rv3d = space.region_3d
                            break
                    break

            if region and rv3d:
                x = event.mouse_x - region.x
                y = event.mouse_y - region.y
                context.window_manager.mouse_pos = (x, y)
                check_axis_hover(context, event, region, rv3d)
                safe_tag_redraw(context)


        # ✅ 左クリックのすべての処理を統合（altあり/なし含む）
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if event.alt:
                # alt + 左クリック → スナップからギズモマトリクス更新
                mat = compute_orientation_matrix_from_snap(context)
                if mat:
                    #print("[INFO] alt + 左クリック → ギズモマトリクスをスナップから更新")
                    temp_gizmo_orientation_matrix = mat
                    safe_tag_redraw(context)
                else:
                    print("[WARN] スナップからマトリクスを取得できませんでした")
                return {'RUNNING_MODAL'}  # ← alt時はパススルーさせない
            elif hover_axis:
                # ギズモ上クリック → 軸選択
                selected_axis = hover_axis
                highlight_locked = True
                safe_tag_redraw(context)
                return {'RUNNING_MODAL'}  # ← ギズモ上でのクリックも内部で消費
            else:
                return {'PASS_THROUGH'}  # ← ギズモ外クリックはBlenderに渡す




        # ✅ alt + 右クリックで現在のトランスフォーム座標系をギズモに反映
        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS' and event.alt:
            obj = context.active_object
            if obj:
                mat = get_orientation_matrix(context, obj)
                #print("[INFO] alt + 右クリック → 現在のトランスフォーム座標系をギズモに反映")
                temp_gizmo_orientation_matrix = mat
                safe_tag_redraw(context)
            else:
                print("[WARN] アクティブオブジェクトが見つかりません")
            return {'RUNNING_MODAL'}  # ★ これがメニュー抑止のキモです




        # ✅ 軸確定後、Enter or Spaceで投影実行
        if event.type in {'RET', 'SPACE'}:
            if not selected_axis:
                self.report({'WARNING'}, "軸を選択してください")
                return {'RUNNING_MODAL'}
            result = project_mesh(context, selected_axis)
            self.cleanup(context)
            return result

        # ✅ ESC or 右クリックで終了
        if event.type == 'ESC':
            self.cleanup(context)
            return {'CANCELLED'}
        elif event.type == 'RIGHTMOUSE' and not event.alt:
            self.cleanup(context)
            return {'CANCELLED'}


        return {'PASS_THROUGH'}

    def cleanup(self, context):
        global gizmo_handler, highlight_handle, help_text_handle, gizmo_showing, highlight_locked

        if gizmo_handler:
            bpy.types.SpaceView3D.draw_handler_remove(gizmo_handler, 'WINDOW')
        if highlight_handle:
            bpy.types.SpaceView3D.draw_handler_remove(highlight_handle, 'WINDOW')
        if help_text_handle:
            bpy.types.SpaceView3D.draw_handler_remove(help_text_handle, 'WINDOW')

        context.scene.transform_orientation_slots[0].type = self.prev_orientation
        if self.custom_orientation_name:
            try:
                bpy.ops.transform.delete_orientation()
            except Exception as e:
                print(f"[WARN] Transform Orientation 削除失敗: {e}")
            self.custom_orientation_name = ""
        self.custom_orientation_name = ""

        unregister_snap_draw_handler()
        snap_draw_state.clear()
        gizmo_handler = highlight_handle = help_text_handle = None
        gizmo_showing = False
        highlight_locked = False

def cast_ray_multiple_directions(bvh, origin, direction, offset_distance=0.01, angle=math.radians(0)):
    # 中央のレイをまずチェック
    hit = bvh.ray_cast(origin, direction)
    if hit and hit[0]:
        return hit

    # 中央の逆方向
    hit = bvh.ray_cast(origin, -direction)
    if hit and hit[0]:
        return hit

    # 角度をつけて周囲に複数発射（ここで「甘い」判定を実現）
    axes = [Vector((1,0,0)), Vector((0,1,0)), Vector((0,0,1))]
    for axis in axes:
        for sign in (-1, 1):
            rot_axis = direction.cross(axis).normalized()
            if rot_axis.length < 0.001:
                continue  # 軸が同じ場合はスキップ
            rot_mat = Matrix.Rotation(angle * sign, 4, rot_axis)
            rotated_dir = (rot_mat @ direction).normalized()
            hit = bvh.ray_cast(origin, rotated_dir)
            if hit and hit[0]:
                return hit
            # 逆方向
            hit = bvh.ray_cast(origin, -rotated_dir)
            if hit and hit[0]:
                return hit

    # オフセットした位置からの追加レイ
    offsets = [
        direction.orthogonal().normalized() * offset_distance,
        -direction.orthogonal().normalized() * offset_distance
    ]
    for offset in offsets:
        offset_origin = origin + offset
        hit = bvh.ray_cast(offset_origin, direction)
        if hit and hit[0]:
            return hit
        hit = bvh.ray_cast(offset_origin, -direction)
        if hit and hit[0]:
            return hit

    return None  # すべて外れた場合



# === 投影処理 ===
def project_mesh(context, axis):
    global cached_edit_coords, cached_edit_verts_indices

    if not cached_edit_objects:
        return {'CANCELLED'}

    was_active = context.view_layer.objects.active
    was_selected = [obj for obj in context.selected_objects]
    depsgraph = context.evaluated_depsgraph_get()
    all_bvhs = []

    # === 投影ターゲット（sourceを含む編集モードオブジェクトも許可） ===
    target_objs = [
        obj for obj in context.view_layer.objects
        if obj.type == 'MESH' and (
            obj.select_get() or obj.mode == 'EDIT'
        )
    ]

    for target in target_objs:
        #print(f"🎯 ターゲット: {target.name}（モード: {target.mode}）")
        try:
            if target.mode == 'EDIT':
                #print("🔧 編集モードターゲット → 選択面から一時オブジェクト作成")

                bm_target = bmesh.from_edit_mesh(target.data)
                bm_target.faces.ensure_lookup_table()
                bm_target.edges.ensure_lookup_table()
                bm_target.verts.ensure_lookup_table()

                selected_faces = [f for f in bm_target.faces if f.select]
                if not selected_faces:
                    selected_verts = {v for v in bm_target.verts if v.select}
                    selected_edges = {e for e in bm_target.edges if e.select}
                    connected_faces = set()
                    for f in bm_target.faces:
                        if any(v in selected_verts for v in f.verts) or any(e in f.edges for e in selected_edges):
                            connected_faces.add(f)
                    selected_faces = list(connected_faces)

                if not selected_faces:
                    #print("⚠ 選択面なし、スキップ")
                    continue

                bm_new = bmesh.new()
                for face in selected_faces:
                    bm_face = bm_new.faces.new([bm_new.verts.new(v.co) for v in face.verts])
                    bm_face.smooth = face.smooth
                bm_new.verts.ensure_lookup_table()
                bm_new.faces.ensure_lookup_table()

                temp_mesh = bpy.data.meshes.new("TempTargetMesh")
                bm_new.to_mesh(temp_mesh)
                bm_new.free()

                temp_obj = bpy.data.objects.new("TempTargetObj", temp_mesh)
                temp_obj.matrix_world = target.matrix_world.copy()
                temp_obj.hide_viewport = True
                temp_obj.hide_render = True
                temp_obj.hide_select = True
                context.scene.collection.objects.link(temp_obj)
                context.view_layer.update()

                bm_eval = bmesh.new()
                bm_eval.from_mesh(temp_mesh)
                bmesh.ops.triangulate(bm_eval, faces=bm_eval.faces)

                verts_world = [temp_obj.matrix_world @ v.co for v in bm_eval.verts]
                tris = [[v.index for v in f.verts] for f in bm_eval.faces if len(f.verts) == 3]
                if tris:
                    all_bvhs.append(BVHTree.FromPolygons(verts_world, tris))
                bm_eval.free()

                bpy.data.objects.remove(temp_obj, do_unlink=True)
                bpy.data.meshes.remove(temp_mesh, do_unlink=True)
                #print("✅ 編集モードターゲット→BVH追加完了")

            else:
                #print("📦 オブジェクトモードターゲット → 通常BVH処理")
                eval_obj = target.evaluated_get(depsgraph)
                bm = bmesh.new()
                bm.from_object(target, depsgraph)
                bmesh.ops.triangulate(bm, faces=bm.faces)
                verts_world = [target.matrix_world @ v.co for v in bm.verts]
                tris = [[v.index for v in f.verts] for f in bm.faces if len(f.verts) == 3]
                if tris:
                    all_bvhs.append(BVHTree.FromPolygons(verts_world, tris))
                bm.free()
                #print("✅ オブジェクトモードターゲット→BVH追加完了")

        except Exception as e:
            print(f"⚠ ターゲット処理失敗 ({target.name}):", e)

    if not all_bvhs:
        #print("⚠ BVHターゲットなし")
        return {'CANCELLED'}

    # === 投影元の頂点を更新（キャッシュ済みの選択頂点） ===
    direction_map = {'X': Vector((1, 0, 0)), 'Y': Vector((0, 1, 0)), 'Z': Vector((0, 0, 1))}
    global temp_gizmo_orientation_matrix

    for source in cached_edit_objects:
        indices = cached_edit_verts_indices.get(source.name, [])
        if not indices:
            continue

        context.view_layer.objects.active = source
        source.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(source.data)
        bm.verts.ensure_lookup_table()

        basis = temp_gizmo_orientation_matrix if temp_gizmo_orientation_matrix else get_orientation_matrix(context, source)
        direction = basis @ direction_map[axis]

        for idx in indices:
            if idx < len(bm.verts):
                v = bm.verts[idx]
                origin_world = source.matrix_world @ v.co
                found_hit = None

                for bvh in all_bvhs:
                    # 現状のまま複数方向からrayを飛ばしてヒット判定
                    found_hit = cast_ray_multiple_directions(bvh, origin_world, direction.normalized())
                    if found_hit and found_hit[0]:
                        hit_point = found_hit[0]

                        # ✅ 修正箇所：移動方向を軸方向だけに制限
                        to_hit_vec = hit_point - origin_world
                        projection_length = to_hit_vec.dot(direction.normalized())
                        new_pos_world = origin_world + direction.normalized() * projection_length

                        v.co = source.matrix_world.inverted() @ new_pos_world
                        break  # 最初に当たったら終了

        bmesh.update_edit_mesh(source.data)



    # === 選択状態の復元 ===
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in was_selected:
        obj.select_set(True)
    context.view_layer.objects.active = was_active

    # ✅ プロジェクション完了後に新たな編集モード状態をキャッシュし直す
    cached_edit_coords = {}
    cached_edit_verts_indices = {}

    for obj in cached_edit_objects:
        context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')  # ← 明示的にEDITモードに戻す

        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        cached_edit_coords[obj.name] = [v.co.copy() for v in bm.verts if v.select]
        cached_edit_verts_indices[obj.name] = [i for i, v in enumerate(bm.verts) if v.select]

    # 最後にモードを戻す（任意で追加）
    bpy.ops.object.mode_set(mode='OBJECT')

    update_highlight_batches()


    return {'FINISHED'}



# === マウストラッカー ===
class ModalMouseTrackerOperator(bpy.types.Operator):
    bl_idname = "wm.modal_mouse_tracker"
    bl_label = "Mouse Tracker"

    def modal(self, context, event):

        if event.alt:
            #print("[DEBUG] alt押下中、スナップハイライト更新")
            update_snap_highlight(context, event)
            safe_tag_redraw(context)
        else:
            # altを離したらハイライト解除
            if snap_draw_state.coords:
                #print("[DEBUG] alt解除 → ハイライト非表示")
                snap_draw_state.clear()
                safe_tag_redraw(context)

        if event.type == 'MOUSEMOVE':
            # VIEW_3D の region と rv3d を手動で取得
            region = None
            rv3d = None
            for area in context.window.screen.areas:
                if area.type == 'VIEW_3D':
                    for reg in area.regions:
                        if reg.type == 'WINDOW':
                            region = reg
                            space = area.spaces.active
                            if space.type == 'VIEW_3D':
                                rv3d = space.region_3d
                            break
                    break

            if not region or not rv3d:
                return {'PASS_THROUGH'}

            # 👇 修正ポイント：マウス座標を region に対して相対的に補正
            x = event.mouse_x - region.x
            y = event.mouse_y - region.y
            context.window_manager.mouse_pos = (x, y)
            #print(f"[TRACKER] MOUSEMOVE adjusted mouse_pos = ({x}, {y})")

            # 正しく補完した region/rv3d を使って hover 判定
            check_axis_hover(context, event, region, rv3d)
            safe_tag_redraw(context)

        if not gizmo_showing:
            return {'FINISHED'}
        return {'PASS_THROUGH'}


    def invoke(self, context, event=None):
        #print("[TRACKER] waiting for MOUSEMOVE to set mouse_pos...")
        context.window_manager.modal_handler_add(self)
        register_snap_draw_handler(context.area)

        return {'RUNNING_MODAL'}


def register_snap_draw_handler(area):
    global snap_draw_handler
    #print("register_snap_draw_handler")
    if area is None:
        # VIEW_3Dエリアを明示的に探す
        for a in bpy.context.window.screen.areas:
            if a.type == 'VIEW_3D':
                area = a
                break
    if snap_draw_handler is None:
        snap_draw_handler = bpy.types.SpaceView3D.draw_handler_add(snap_draw_state.draw_callback, (), 'WINDOW', 'POST_VIEW')
    if area:
        area.tag_redraw()

def unregister_snap_draw_handler():
    global snap_draw_handler
    if snap_draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(snap_draw_handler, 'WINDOW')
        snap_draw_handler = None



# === 登録処理 ===
addon_keymaps = []

def register():
    bpy.utils.register_class(ModalMeshProjectorOperator)
    bpy.utils.register_class(ModalMouseTrackerOperator)
    bpy.utils.register_class(LaunchMeshProjectorOperator) 
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.append(draw_mesh_projector_menu)
    bpy.types.WindowManager.mouse_pos = IntVectorProperty(name="Mouse Pos", size=2, default=(0, 0))
    #wm = bpy.context.window_manager
    #km = wm.keyconfigs.addon.keymaps.new(name="3D View", space_type='VIEW_3D')
    #kmi = km.keymap_items.new("view3d.modal_mesh_projector", type='EIGHT', value='PRESS')
    #addon_keymaps.append((km, kmi))

def unregister():
    global gizmo_handler
    if gizmo_handler:
        bpy.types.SpaceView3D.draw_handler_remove(gizmo_handler, 'WINDOW')
        gizmo_handler = None
    #for km, kmi in addon_keymaps:
        #km.keymap_items.remove(kmi)
    #addon_keymaps.clear()
    if hasattr(bpy.types.WindowManager, "mouse_pos"):
        del bpy.types.WindowManager.mouse_pos
    bpy.utils.unregister_class(ModalMeshProjectorOperator)
    bpy.utils.unregister_class(ModalMouseTrackerOperator)
    bpy.utils.unregister_class(LaunchMeshProjectorOperator)
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.remove(draw_mesh_projector_menu)

if __name__ == "__main__":
    register()