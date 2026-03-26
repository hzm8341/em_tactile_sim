import numpy as np
import mujoco
from ..core.sensor_config import SensorConfig
from ..core.contact_model import ContactModel
from ..core import hall_response


class EMSensorCallback:
    """
    Registers a MuJoCo mjcb_sensor callback that computes the EM_SENSOR output
    using Hertz contact mechanics every physics step.
    """

    def __init__(self, model: mujoco.MjModel, config: SensorConfig) -> None:
        self._cfg = config
        self._contact_model = ContactModel(config)

        # Resolve IDs once at construction
        self._sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "EM_SENSOR")
        if self._sensor_id < 0:
            raise ValueError("Sensor 'EM_SENSOR' not found in model.")
        self._sensor_adr = int(model.sensor_adr[self._sensor_id])

        actual_dim = int(model.sensor_dim[self._sensor_id])
        if actual_dim != self._cfg.sensor_dim:
            raise ValueError(
                f"XML sensor dim {actual_dim} != config sensor_dim {self._cfg.sensor_dim}")

        self._pad_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "sensor_pad")
        if self._pad_geom_id < 0:
            raise ValueError("Geom 'sensor_pad' not found in model.")

    def register(self) -> None:
        """Register this callback with MuJoCo. Call once after MjData is created."""
        mujoco.set_mjcb_sensor(self._callback)

    def unregister(self) -> None:
        """Remove the callback (e.g. on env teardown)."""
        mujoco.set_mjcb_sensor(None)

    def _callback(self, model: mujoco.MjModel,
                        data: mujoco.MjData,
                        stage: int) -> None:
        # mjcb_sensor is called at multiple stages; sensor data filled at ACC stage
        if stage != mujoco.mjtStage.mjSTAGE_ACC:
            return

        contacts = self._get_pad_contacts(model, data)
        force_array = self._contact_model.compute_multi(contacts)
        output = hall_response.compute_output(force_array, self._cfg)

        adr = self._sensor_adr
        data.sensordata[adr: adr + self._cfg.sensor_dim] = output

    def _get_pad_contacts(self, model: mujoco.MjModel,
                                data: mujoco.MjData) -> list:
        """Extract contacts involving sensor_pad, converted to pad-local coords."""
        pad_id = self._pad_geom_id
        pad_pos = data.geom_xpos[pad_id].copy()          # world position of geom
        pad_mat = data.geom_xmat[pad_id].copy().reshape(3, 3)  # rotation matrix (world←geom)

        contact_force = np.zeros(6, dtype=np.float64)
        contacts = []

        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 != pad_id and c.geom2 != pad_id:
                continue

            mujoco.mj_contactForce(model, data, i, contact_force)
            fn  = float(contact_force[0])   # always positive for compression
            ftx = float(contact_force[1])
            fty = float(contact_force[2])

            if fn < self._cfg.sensitivity:
                continue

            # Contact position → pad-local XY
            world_pos = c.pos.copy()
            local_3d  = pad_mat.T @ (world_pos - pad_pos)

            other_geom_id = int(c.geom2) if int(c.geom1) == pad_id else int(c.geom1)
            radius = self._get_geom_radius(model, other_geom_id)

            contacts.append({
                "pos":    local_3d[:2],
                "fn":     fn,
                "ft":     np.array([ftx, fty]),
                "radius": radius,
            })

        return contacts

    @staticmethod
    def _get_geom_radius(model: mujoco.MjModel, geom_id: int) -> float:
        """Infer equivalent Hertz radius from geom type and size."""
        gtype = int(model.geom_type[geom_id])
        size  = model.geom_size[geom_id]

        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(size[0])
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return float(size[0])   # end-cap radius
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            return float(np.min(size[:3]))
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(size[0])
        else:
            return 0.01             # fallback default
